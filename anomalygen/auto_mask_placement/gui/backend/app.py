# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integrated Flask Backend for Automatic Mask Placement & ROI Generation
"""

import base64
import hashlib
import json
import logging
import os
import pathlib
import random as rand_mod
import re
import secrets
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from dataclasses import replace
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import torch
from cosmos_framework.utils import log
from flask import Flask, g, jsonify, redirect, request, send_file, send_from_directory
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageOps
from werkzeug.local import LocalProxy
from werkzeug.utils import secure_filename

from anomalygen.auto_mask_placement import AlignmentPoint, AugmentationParams, AutoMaskPlacement
from anomalygen.auto_mask_placement.roi_generation.default_config import ROIGenerationConfig
from anomalygen.auto_mask_placement.roi_generation.model import ROIGenerationModels
from anomalygen.auto_mask_placement.roi_generation.pipeline import run_roi_pipeline

_REDACTED = "[redacted]"
_ERR_NO_FILE = "No file provided"
_ERR_FILE_TYPE = "Invalid file type"
_ROI_VISUALIZATION_PNG = "roi_visualization.png"

app = Flask(__name__, static_folder="../frontend/public", static_url_path="")

# No CORS. This app serves its own UI from static_folder above, and the page calls the API with a
# relative base ("/api"), so every legitimate request is same-origin and needs no CORS header.

# Configuration
# Per-session temp roots. Each session writes under its own <base>/<sid>/
# subdirectory so concurrent browsers never collide on fixed filenames
# (submask_resized.png, drawn_rois.json, seed_N/, ...).
_UPLOAD_BASE = tempfile.mkdtemp(prefix="amp_gui_uploads_")
_OUTPUT_BASE = tempfile.mkdtemp(prefix="amp_gui_outputs_")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "json", "bmp", "webp"}

# Roots a custom output directory may live under. /api/output-directory/set takes a path from the
# request, so without a boundary it is an arbitrary-write primitive: anything the user account can
# write, including this app's own static assets, shell rc files, or ~/.ssh.
#
# The default root is the repo checkout, derived from this file rather than the working directory:
# the documented way to start the server is from the checkout, but a unit file with no
# WorkingDirectory would otherwise make the boundary "/" and confine nothing. AMP_OUTPUT_ROOT
# widens it for a deployment that keeps outputs elsewhere. The temp base stays allowed because it
# is where sessions write by default.
#
# What this does and does not stop: paths outside the root — ~/.ssh, shell rc files, /etc — are
# rejected. Paths inside it are not, so this app's own static assets are still writable by a caller
# that can reach the endpoint. os.makedirs only creates directories, so the worst case in-repo is
# the pipeline writing outputs somewhere odd, not code being overwritten.
_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_OUTPUT_ROOT = os.path.realpath(os.environ.get("AMP_OUTPUT_ROOT", _REPO_ROOT))
_ALLOWED_OUTPUT_ROOTS = (_OUTPUT_ROOT, os.path.realpath(_OUTPUT_BASE))


def _resolve_output_directory(requested: str) -> str:
    """Resolve a requested output directory, or raise ValueError if it escapes the allowed roots.

    Resolves symlinks before comparing: a symlink inside an allowed root pointing outside it would
    otherwise pass a prefix check and still write anywhere.
    """
    candidate = os.path.realpath(os.path.expanduser(requested))
    for root in _ALLOWED_OUTPUT_ROOTS:
        if candidate == root or candidate.startswith(root + os.sep):
            return candidate
    raise ValueError(f"output directory must be inside {_OUTPUT_ROOT} (set AMP_OUTPUT_ROOT to allow another location)")


app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Per-session state, keyed by a stable id the frontend sends (X-AMP-Session
# header, else the peer address as a last resort). `session_data` is a LocalProxy that
# resolves to the current request's dict, so existing `session_data[...]` usage
# is unchanged but isolated per browser instead of shared across all clients.
_sessions = {}
_sessions_lock = threading.Lock()


# The session id becomes a directory name under the upload and output roots, so it must be
# one safe path segment — the realpath boundary below guards the request's directory fields,
# not this. Unsafe values are hashed rather than rejected, so an unusual but legitimate id
# (an IPv6 peer address, the fallback) still gets its own directory.
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _session_key(raw: str) -> str:
    """Reduce a client-supplied session id to one safe path segment."""
    if _SAFE_SESSION_ID.fullmatch(raw) and raw not in (".", ".."):
        return raw
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:32]


def _current_session():
    sess = getattr(g, "_amp_session", None)
    if sess is None:
        sid = _session_key(request.headers.get("X-AMP-Session") or request.remote_addr or "default")
        with _sessions_lock:
            sess = _sessions.setdefault(sid, {})
            sess.setdefault("_sid", sid)
        g._amp_session = sess
    return sess


session_data = LocalProxy(_current_session)


# "Only reachable on loopback" is not an access control: every process on the host can
# call these routes. A token minted at startup and handed to the browser as a
# SameSite=Strict cookie closes two gaps at once — another local process cannot guess
# it, and a foreign page cannot make the browser send it, which is the CSRF defence.
# AMP_AUTH_TOKEN pins the value for a scripted or containerised launch.
_AUTH_TOKEN = os.environ.get("AMP_AUTH_TOKEN") or secrets.token_urlsafe(32)
_TOKEN_COOKIE = "amp_token"
# Readiness polling starts before the page has the cookie, and the reply is booleans only.
_UNAUTHENTICATED_PATHS = frozenset({"/api/health"})


def _token_matches(supplied: str) -> bool:
    """Constant-time check against the launch token.

    .isascii() first: header, cookie and query values all decode as latin-1, and
    compare_digest raises TypeError on a non-ASCII str, which Flask would surface as a 500
    rather than a refusal. The token is URL-safe base64, so a non-ASCII value could never
    match. One helper because the token arrives from three different inputs.
    """
    return bool(supplied) and supplied.isascii() and secrets.compare_digest(supplied, _AUTH_TOKEN)


@app.before_request
def _require_local_token():
    """Gate /api/* on the launch token. Static assets carry no privilege and stay open."""
    if not request.path.startswith("/api/") or request.path in _UNAUTHENTICATED_PATHS:
        return None
    supplied = request.headers.get("X-AMP-Token") or request.cookies.get(_TOKEN_COOKIE, "")
    if not _token_matches(supplied):
        return jsonify({"error": "missing or invalid token — open the URL printed by the launcher"}), 401
    return None


# Most handlers return ``str(e)`` to the caller, and an upstream client error can carry a
# credential in its text (an HF URL, a token in a message). Scrubbing on the way out covers
# the handlers that exist and the ones added later. Errors only: success bodies carry base64
# imagery and are not worth scanning.
_SECRET_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}")


@app.after_request
def _redact_secrets_from_errors(response):
    if response.status_code < 400 or not response.is_json:
        return response
    body = response.get_data(as_text=True)
    redacted = _SECRET_PATTERN.sub(_REDACTED, body)
    if redacted != body:
        response.set_data(redacted)
    return response


# Werkzeug logs the request line, so the one-time ``GET /?token=...`` exchange would put the
# token into whatever sink those logs reach — a file, a terminal, an aggregator. The terminal
# already shows it once by design; a retained log is a different lifetime.
_TOKEN_IN_URL = re.compile(r"(?<=token=)[^\s&\"]+")


class _RedactTokenInAccessLog(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = _TOKEN_IN_URL.sub(_REDACTED, record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_TOKEN_IN_URL.sub(_REDACTED, a) if isinstance(a, str) else a for a in record.args)
        return True


logging.getLogger("werkzeug").addFilter(_RedactTokenInAccessLog())


def upload_folder():
    """Current session's upload directory (created on demand)."""
    d = os.path.join(_UPLOAD_BASE, session_data["_sid"])
    os.makedirs(d, exist_ok=True)
    return d


def output_folder():
    """Current session's output directory, or its custom override."""
    override = session_data.get("_output_override")
    d = override if override else os.path.join(_OUTPUT_BASE, session_data["_sid"])
    os.makedirs(d, exist_ok=True)
    return d


# ROI Generation Globals
cached_roi_models = None
roi_device = None
roi_init_error = None
roi_init_in_progress = False

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


def allowed_file(filename):
    return pathlib.Path(filename).suffix.lower()[1:] in ALLOWED_EXTENSIONS


def get_default_roi_generation_config():
    """
    Get the default filter configuration template.
    This ensures frontend and backend always use the same parameter structure.

    Returns:
        dict: Flattened config with all filter parameters
    """
    cfg = OmegaConf.structured(ROIGenerationConfig)
    return {
        # Method enable flags
        "template_box_to_masks_enabled": cfg.template_box_to_masks.enabled,
        "box_to_mask_enabled": cfg.box_to_mask.enabled,
        "grayscale_to_mask_enabled": cfg.grayscale_to_mask.enabled,
        # Template parameters
        "max_template_boxes": cfg.template_box_to_masks.max_template,
        "crop_resize": cfg.template_box_to_masks.crop_resize,
        "refine_template": cfg.template_box_to_masks.refine_template,
        "rotation_degrees_text": ", ".join(str(v) for v in cfg.template_box_to_masks.rotation_degrees),
        "rotation_degrees": list(cfg.template_box_to_masks.rotation_degrees),
        "allow_flip": list(cfg.template_box_to_masks.allow_flip),
        # Proposal parameters
        "proposal_similarity_tol": cfg.template_box_to_masks.proposal_similarity_tol,
        "max_proposal": cfg.template_box_to_masks.max_proposal,
        # Box filter parameters
        "box_enabled": cfg.template_box_to_masks.box_enabled,
        "size_tol": cfg.template_box_to_masks.size_tol,
        "aspect_tol": cfg.template_box_to_masks.aspect_tol,
        # Mask filter parameters
        "mask_enabled": cfg.template_box_to_masks.mask_enabled,
        "component_count_tol": cfg.template_box_to_masks.component_count_tol,
        "chamfer_tol": cfg.template_box_to_masks.chamfer_tol,
        # HOG filter parameters
        "hog_enabled": cfg.template_box_to_masks.hog_enabled,
        "hog_similarity_tol": cfg.template_box_to_masks.hog_similarity_tol,
        # Color histogram filter parameters
        "color_hist_enabled": cfg.template_box_to_masks.color_hist_enabled,
        "lightness_tol": cfg.template_box_to_masks.lightness_tol,
        "color_tol": cfg.template_box_to_masks.color_tol,
        # Grayscale to mask parameters
        "grayscale_to_mask_threshold_mode": cfg.grayscale_to_mask.threshold_mode,
        "grayscale_to_mask_threshold_value": cfg.grayscale_to_mask.threshold_value,
        # Post-processing parameters
        "morphological_operation": cfg.morphological_operation,
        "morphological_kernel": list(cfg.morphological_kernel),
    }


def merge_config_with_defaults(user_config):
    """
    Merge user-provided config with default config template.
    Ensures all required parameters exist with valid values.

    Args:
        user_config (dict): User-provided configuration (may be partial or empty)

    Returns:
        dict: Complete configuration with all parameters
    """
    # Start with defaults
    config = get_default_roi_generation_config()

    # Override with user values if provided
    if user_config:
        for key, value in user_config.items():
            if key in config:
                # Validate type matches (basic validation)
                default_value = config[key]
                if default_value is not None and value is not None:
                    if isinstance(default_value, bool):
                        config[key] = bool(value)
                    elif isinstance(default_value, int):
                        config[key] = int(value)
                    elif isinstance(default_value, float):
                        config[key] = float(value)
                    else:
                        config[key] = value
                else:
                    config[key] = value
            else:
                # Allow extra keys (for future extensibility)
                config[key] = value

    return config


def process_stage_result_of_template_box_to_masks(result_json_path, config, image, binary_mask, bboxes):
    """
    Process result.json and create response with previews and stage results.

    Args:
        result_json_path: Path to result.json file
        config: Configuration dict with filter parameters
        image: PIL Image object
        binary_mask: Binary mask numpy array (H, W) with values 0-255
        bboxes: List of input bbox dicts

    Returns:
        dict: Response dictionary with masks, previews, stage results, etc.
    """
    img_width, img_height = image.size

    # Process result.json to get stage results and create previews
    stage_results = {}
    stage_images = {}

    # Calculate scale factor if image was resized during processing
    scale_factor = 1.0

    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, "r") as f:
                result_data = json.load(f)

            # Calculate scale factor from image dimensions
            # The ROI generation pipeline resizes images to max 1024px on longest edge by default
            if "image_path" in result_data and os.path.exists(result_data["image_path"]):
                try:
                    # Load the processed image to get its ORIGINAL size (before our resize)
                    processed_img = Image.open(result_data["image_path"])
                    # Apply EXIF orientation to get true dimensions
                    try:
                        processed_img = ImageOps.exif_transpose(processed_img)
                    except Exception:
                        pass
                    processed_width, processed_height = processed_img.size

                    # The pipeline resizes to max 1024px
                    max_resize = 1024
                    processed_max_dim = max(processed_width, processed_height)

                    if processed_max_dim > max_resize:
                        # Image was resized during processing
                        # Scale factor = original_size / resized_size
                        scale_factor = processed_max_dim / max_resize
                        log.info(f"[Process Template-Box-to-Masks Result] Calculated scale factor: {scale_factor:.3f}")
                    else:
                        log.info(
                            f"[Process Template-Box-to-Masks Result] Image was not resized (max_dim {processed_max_dim} <= {max_resize})"
                        )
                except Exception as e:
                    log.error(
                        f"[Process Template-Box-to-Masks Result] Could not load processed image for scale calculation: {e}"
                    )

            # Fallback: use current image dimensions
            scale_factor = _fallback_scale_factor(scale_factor, img_width, img_height)

            # Get candidate boxes and filter metrics
            candidate_boxes = np.array(result_data["candidate_boxes"])

            if len(candidate_boxes) > 0:
                size_diff = np.array(result_data["size_diff"])
                aspect_diff = np.array(result_data["aspect_diff"])
                component_diffs = np.array(result_data["component_diffs"])
                chamfer_score = np.array(result_data["chamfer_score"])
                sim_hog = np.array(result_data["sim_hog"])
                sim_lightness = np.array(result_data["sim_lightness"])
                sim_color = np.array(result_data["sim_color"])

                # Stage 1: Proposal (all candidates)
                proposal_boxes = candidate_boxes

                # Stage 2: Box filter
                if config.get("box_enabled", True):
                    keep_box_size = size_diff <= config.get("size_tol", 0.3)
                    keep_box_aspect = aspect_diff <= config.get("aspect_tol", 0.3)
                    keep_box = keep_box_size & keep_box_aspect
                else:
                    keep_box = np.ones(len(candidate_boxes), dtype=bool)
                box_filter_boxes = candidate_boxes[keep_box]

                # Stage 3: Mask filter
                if config.get("mask_enabled", True):
                    keep_mask_component = component_diffs <= config.get("component_count_tol", 0)
                    keep_mask_chamfer = chamfer_score <= config.get("chamfer_tol", 0.05)
                    keep_mask = keep_box & keep_mask_component & keep_mask_chamfer
                else:
                    keep_mask = keep_box
                mask_filter_boxes = candidate_boxes[keep_mask]

                # Stage 4a: HOG filter
                if config.get("hog_enabled", True):
                    keep_hog = keep_mask & (sim_hog >= (1 - config.get("hog_similarity_tol", 0.8)))
                else:
                    keep_hog = keep_mask
                hog_filter_boxes = candidate_boxes[keep_hog]

                # Stage 4b: Color histogram filter
                if config.get("color_hist_enabled", True):
                    keep_color = (
                        keep_hog
                        & (sim_lightness >= (1 - config.get("lightness_tol", 0.5)))
                        & (sim_color >= (1 - config.get("color_tol", 0.5)))
                    )
                else:
                    keep_color = keep_hog
                color_filter_boxes = candidate_boxes[keep_color]

                # Create stage data - KEY FIX: Use the keys expected by frontend
                stage_data = {
                    "Template-Box-to-Masks_proposal": proposal_boxes,
                    "Template-Box-to-Masks_box_filter": box_filter_boxes,
                    "Template-Box-to-Masks_mask_filter": mask_filter_boxes,
                    "Template-Box-to-Masks_hog_filter": hog_filter_boxes,
                    "Template-Box-to-Masks_color_filter": color_filter_boxes,
                }

                # Add template stage
                template_boxes = None
                is_refined_template = False

                if "refined_template_boxes" in result_data:
                    template_boxes = np.array(result_data["refined_template_boxes"])
                    is_refined_template = True
                    log.info(
                        f"[Process Template-Box-to-Masks Result] Using {len(template_boxes)} refined template boxes from result"
                    )
                elif bboxes:
                    template_boxes = np.array(
                        [
                            [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
                            for bbox in bboxes
                        ]
                    )

                # Create stage results and preview images
                for stage_key, stage_boxes in stage_data.items():
                    # Store original boxes (in resized coordinates) for result
                    stage_results[stage_key] = {
                        "count": len(stage_boxes),
                        "enabled": True,
                        "boxes": stage_boxes.tolist() if isinstance(stage_boxes, np.ndarray) else stage_boxes,
                    }
                    log.info(f"[Process Result] {stage_key}: {len(stage_boxes)} boxes")

                    if len(stage_boxes) > 0:
                        # Scale boxes back to original image coordinates for visualization
                        # (except template boxes which are already in original coordinates unless refined/resized)
                        boxes_for_viz = stage_boxes

                        should_scale = scale_factor > 1.0
                        if stage_key == "template" and not is_refined_template:
                            should_scale = False

                        if should_scale:
                            boxes_for_viz = stage_boxes * scale_factor
                            log.info(
                                f"[Process Result] Scaled {stage_key} boxes by {scale_factor:.3f} for visualization"
                            )

                        stage_bboxes = [
                            {
                                "x": box[0],
                                "y": box[1],
                                "width": box[2] - box[0],
                                "height": box[3] - box[1],
                                "id": f"{stage_key}_{i}",
                            }
                            for i, box in enumerate(boxes_for_viz)
                            if len(box) == 4
                        ]

                        stage_images[stage_key] = create_overlay_image(image, bboxes=stage_bboxes, masks=None)
                        log.info(f"[Process Template-Box-to-Masks Result] Created preview for {stage_key}")

        except Exception as e:
            log.error(f"[Process Template-Box-to-Masks Result] Error computing stage boxes from result.json: {e}")
            traceback.print_exc()

    # Create final stage: bounding boxes of connected components from final binary mask
    num_labels, labels = cv2.connectedComponents(binary_mask)
    if num_labels > 1:
        final_boxes = []
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            # Find bounding box of this connected component
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                final_boxes.append([float(x), float(y), float(x + w), float(y + h)])

        if len(final_boxes) > 0:
            final_boxes = np.array(final_boxes)
            stage_results["Template-Box-to-Masks_final"] = {
                "count": len(final_boxes),
                "enabled": True,
                "boxes": final_boxes.tolist(),
            }

            # Create preview with final boxes
            final_bboxes = [
                {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1], "id": f"final_{i}"}
                for i, box in enumerate(final_boxes)
            ]

            stage_images["Template-Box-to-Masks_final"] = create_overlay_image(image, bboxes=final_bboxes, masks=None)
            log.info(
                f"[Process Template-Box-to-Masks Result] final: {len(final_boxes)} boxes (bounding boxes of connected components)"
            )

    results = {
        "stageResults": stage_results,
        "stageImages": stage_images,
    }

    return results


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _intermediate_images(output_dir):
    """Base64 of the first seed's intermediate images, by result key. A missing file is left out."""
    names = {
        "roi_visualization": _ROI_VISUALIZATION_PNG,
        "cropped_submask": "cropped_submask.png",
        "combined_rois_binary": "combined_rois_binary.png",
    }
    paths = {key: os.path.join(output_dir, name) for key, name in names.items()}
    return {key: image_to_base64(path) for key, path in paths.items() if os.path.exists(path)}


def _fallback_scale_factor(scale_factor, img_width, img_height):
    """Keep a scale factor found from the processed image. While it is still the 1.0 default, estimate
    one from the current image against the pipeline's 1024px resize limit."""
    max_resize = 1024
    original_max_dim = max(img_width, img_height)
    if scale_factor > 1.0 or original_max_dim <= max_resize:
        return scale_factor
    scale_factor = original_max_dim / max_resize
    log.info(
        f"[Process Template-Box-to-Masks Result] Fallback: Estimated scale factor from current image: {scale_factor:.3f} ({img_width}x{img_height})"
    )
    return scale_factor


def array_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def binary_mask_to_base64(mask_array):
    """Convert a binary mask numpy array to base64 encoded PNG"""
    return array_to_base64(mask_array)


def create_overlay_image(image, bboxes=None, masks=None):
    """Create an overlay visualization with bboxes and masks"""
    img = image.copy()
    if img.mode != "RGB" and img.mode != "RGBA":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0, 200), width=3)

    if masks:
        colors = [(0, 255, 0, 100), (0, 0, 255, 100), (255, 255, 0, 100), (255, 0, 255, 100), (0, 255, 255, 100)]
        for i, mask in enumerate(masks):
            points = mask.get("points", [])
            if points:
                point_tuples = [(p[0], p[1]) for p in points]
                color = colors[i % len(colors)]
                draw.polygon(point_tuples, fill=color, outline=(0, 255, 0, 255), width=2)

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()


# -------------------------------------------------------------------------
# ROI Model Initialization
# -------------------------------------------------------------------------


def initialize_roi_models():
    """Initialize ROI generation models in background"""
    global cached_roi_models, roi_device, roi_init_error, roi_init_in_progress

    if cached_roi_models is not None:
        return

    if roi_init_in_progress:
        return

    roi_init_in_progress = True
    log.info(f"[INIT] Initializing ROI models...")

    try:
        roi_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cached_roi_models = ROIGenerationModels(roi_device)
        log.info(f"[INIT] ROI Models initialized successfully on {roi_device}")
    except Exception as e:
        roi_init_error = str(e)
        log.info(f"[INIT] ROI Model initialization failed: {e}")
        traceback.print_exc()
    finally:
        roi_init_in_progress = False


def background_init():
    try:
        initialize_roi_models()
    except Exception as e:
        log.info(f"[INIT] Background initialization exception: {e}")


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------


@app.route("/")
def serve_frontend():
    """Serve the UI, exchanging ?token=... for the cookie every later /api call carries.

    A successful exchange redirects to ``/`` rather than rendering in place. Access logs are
    already scrubbed above, but the URL itself outlives the request: it lands in browser history
    and in the address bar, and would be sent as a ``Referer`` on any cross-origin subresource the
    page loads. Redirecting spends the token once and leaves a clean URL holding the cookie, which
    is the credential from then on. ``Referrer-Policy`` covers the one hop before the redirect.
    """
    if _token_matches(request.args.get("token", "")):
        response = redirect("/", code=303)
        # secure= follows the transport: hardcoding it would drop the cookie, and with it every
        # later /api call, on the plain-HTTP loopback bind this normally runs on.
        response.set_cookie(
            _TOKEN_COOKIE, _AUTH_TOKEN, httponly=True, samesite="Strict", path="/", secure=request.is_secure
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
    response = send_from_directory(app.static_folder, "index.html")
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.route("/api/health", methods=["GET"])
def health_check():
    status = {
        "status": "ok",
        "message": "Server is running",
        "roi_models_ready": cached_roi_models is not None,
        "roi_init_failed": roi_init_error is not None,
        "roi_init_in_progress": roi_init_in_progress,
    }
    return jsonify(status)


@app.route("/api/session/state", methods=["GET"])
def get_session_state():
    """Get current session state (images, ROIs) to restore frontend on refresh"""
    try:
        state = {
            "input_image": None,
            "submask": None,
            "roi_legal_images": [],
            "roi_illegal_images": [],
            "stage_images": {},
            "heatmap": None,
        }

        if "input_image" in session_data and os.path.exists(session_data["input_image"]):
            state["input_image"] = {
                "url": image_to_base64(session_data["input_image"]),
                "width": session_data.get("image_width"),
                "height": session_data.get("image_height"),
            }

        if "submask" in session_data and os.path.exists(session_data["submask"]):
            state["submask"] = image_to_base64(session_data["submask"])

        # ROI Images
        for key, target in [("roi_legal_images", "roi_legal_images"), ("roi_illegal_images", "roi_illegal_images")]:
            if key in session_data:
                valid_images = []
                for filepath in session_data[key]:
                    if os.path.exists(filepath):
                        # Reconstruct preview object
                        filename = os.path.basename(filepath)
                        # Try to parse timestamp from filename if possible, else use file mtime
                        timestamp = int(os.path.getmtime(filepath))
                        try:
                            # Format: name_timestamp.ext
                            name, ext = os.path.splitext(filename)
                            parts = name.rsplit("_", 1)
                            if len(parts) > 1 and parts[1].isdigit():
                                timestamp = int(parts[1])
                        except Exception:
                            pass

                        # Read preview
                        try:
                            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                preview_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                                is_legal = target == "roi_legal_images"
                                if is_legal:
                                    preview_img[img == 255] = [255, 255, 255]
                                else:
                                    preview_img[img == 255] = [255, 0, 0]

                                valid_images.append(
                                    {
                                        "filepath": filepath,
                                        "filename": filename,
                                        "timestamp": timestamp,
                                        "roi_count": 1,
                                        "image_size": {
                                            "width": session_data.get("image_width", 0),
                                            "height": session_data.get("image_height", 0),
                                        },
                                        "preview": array_to_base64(preview_img),
                                        "is_legal": is_legal,
                                    }
                                )
                        except Exception as e:
                            log.info(f"Error reading {filepath}: {e}")

                state[target] = valid_images

        if "stage_images" in session_data:
            state["stage_images"] = session_data["stage_images"]

        return jsonify(state)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/output-directory/get", methods=["GET"])
def get_output_directory():
    """Get current output directory path"""
    try:
        return jsonify({"output_directory": output_folder(), "is_temp": session_data.get("_is_temp_output", True)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/output-directory/set", methods=["POST"])
def set_output_directory():
    """Set custom output directory"""
    try:
        data = request.json
        custom_dir = data.get("directory", "").strip()

        if not custom_dir:
            return jsonify({"error": "Directory path is required"}), 400

        # Validate before creating: makedirs on an unchecked path is itself the write primitive.
        try:
            custom_dir = _resolve_output_directory(custom_dir)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        os.makedirs(custom_dir, exist_ok=True)

        session_data["_output_override"] = custom_dir
        session_data["_is_temp_output"] = False

        return jsonify(
            {
                "success": True,
                "message": f"Output directory set to: {custom_dir}",
                "output_directory": custom_dir,
                "is_temp": False,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/output-directory/reset", methods=["POST"])
def reset_output_directory():
    """Reset to temporary output directory"""
    try:
        session_data.pop("_output_override", None)
        session_data["_is_temp_output"] = True
        new_temp_dir = output_folder()

        return jsonify(
            {
                "success": True,
                "message": f"Reset to temporary directory: {new_temp_dir}",
                "output_directory": new_temp_dir,
                "is_temp": True,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- AMP Endpoints ---


@app.route("/api/upload/input-image", methods=["POST"])
def upload_input_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": _ERR_NO_FILE}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder(), f"input_{filename}")
            file.save(filepath)
            session_data["ori_filename"] = filename

            # Handle EXIF orientation
            try:
                img = Image.open(filepath)
                img = ImageOps.exif_transpose(img)
                img.save(filepath)
            except Exception:
                img = Image.open(filepath)

            width, height = img.size

            session_data["input_image"] = filepath
            session_data["image_width"] = width
            session_data["image_height"] = height

            return jsonify(
                {
                    "success": True,
                    "filepath": filepath,
                    "width": width,
                    "height": height,
                    "image": image_to_base64(filepath),
                }
            )

        return jsonify({"error": _ERR_FILE_TYPE}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload/submask", methods=["POST"])
def upload_submask():
    try:
        if "file" not in request.files:
            return jsonify({"error": _ERR_NO_FILE}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder(), f"submask_{filename}")
            file.save(filepath)

            img = Image.open(filepath)
            width, height = img.size
            session_data["submask"] = filepath
            # Clear pool mode when uploading single file
            session_data.pop("submask_pool_path", None)
            session_data.pop("submask_pool_files", None)

            return jsonify(
                {
                    "success": True,
                    "filepath": filepath,
                    "width": width,
                    "height": height,
                    "image": image_to_base64(filepath),
                }
            )
        return jsonify({"error": _ERR_FILE_TYPE}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browse/submask-folder", methods=["POST"])
def browse_submask_folder():
    """Browse a folder of submask images uploaded from the client."""
    try:
        if "files[]" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files[]")
        candidates = []

        for file in files:
            if file and allowed_file(file.filename):
                try:
                    img = Image.open(file.stream)
                    # Convert to binary for preview
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    candidates.append(
                        {"filename": file.filename, "image": img_base64, "width": img.size[0], "height": img.size[1]}
                    )
                except Exception:
                    continue

        if not candidates:
            return jsonify({"error": "No valid image files found"}), 400

        session_data["submask_candidates"] = candidates
        return jsonify({"success": True, "total_count": len(candidates), "candidates": candidates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/select/submask", methods=["POST"])
def select_submask():
    """Select a submask from the browsed candidates."""
    try:
        data = request.json
        index = data.get("index", 0)

        if "submask_candidates" not in session_data:
            return jsonify({"error": "No submask candidates available"}), 400

        candidates = session_data["submask_candidates"]
        if index < 0 or index >= len(candidates):
            return jsonify({"error": "Invalid index"}), 400

        selected = candidates[index]

        # Save the selected submask to a file
        img_data = base64.b64decode(selected["image"])
        # secure_filename, like every other upload handler here: this name came from the client and
        # is stored verbatim by /api/browse/submask-folder, so a crafted "../../x.png" would walk out
        # of the upload folder with attacker-controlled bytes. The prefix does not prevent that.
        filepath = os.path.join(upload_folder(), f"submask_selected_{secure_filename(selected['filename'])}")
        with open(filepath, "wb") as f:
            f.write(img_data)

        session_data["submask"] = filepath
        # Clear pool mode when selecting single file
        session_data.pop("submask_pool_path", None)
        session_data.pop("submask_pool_files", None)

        return jsonify(
            {"success": True, "filepath": filepath, "width": selected["width"], "height": selected["height"]}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/validate/submask-pool", methods=["POST"])
def validate_submask_pool():
    """Validate a server-side directory as a submask pool."""
    try:
        data = request.json
        pool_path = data.get("path", "").strip()

        if not pool_path:
            return jsonify({"error": "Path is required"}), 400

        # Same boundary as the output directory: this path comes from the request and the response
        # returns a base64 image read from it, so without a root it is an arbitrary directory
        # listing plus an arbitrary image read. Pools outside the checkout need AMP_OUTPUT_ROOT.
        try:
            pool_path = _resolve_output_directory(pool_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if not os.path.isdir(pool_path):
            return jsonify({"error": f"Directory not found: {pool_path}"}), 400

        # Find all image files in the directory
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        image_files = []
        for fname in os.listdir(pool_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(pool_path, fname))

        if not image_files:
            return jsonify({"error": "No image files found in directory"}), 400

        # Get a random sample for preview
        sample_path = rand_mod.choice(image_files)
        sample_image = image_to_base64(sample_path)

        # Store pool path in session
        session_data["submask_pool_path"] = pool_path
        session_data["submask_pool_files"] = image_files

        # Also set the sample as current submask for preview
        session_data["submask"] = sample_path

        return jsonify(
            {"success": True, "count": len(image_files), "sample_image": sample_image, "sample_path": sample_path}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload/roi-json", methods=["POST"])
def upload_roi_json():
    try:
        if "file" not in request.files:
            return jsonify({"error": _ERR_NO_FILE}), 400

        file = request.files["file"]
        if file and file.filename.endswith(".json"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder(), f"roi_{filename}")
            file.save(filepath)

            with open(filepath, "r") as f:
                roi_data = json.load(f)

            session_data["roi_json"] = filepath

            legal_count = sum(1 for roi in roi_data.get("rois", []) if roi.get("is_legal", True))
            total_count = len(roi_data.get("rois", []))

            return jsonify(
                {
                    "success": True,
                    "filepath": filepath,
                    "total_rois": total_count,
                    "legal_rois": legal_count,
                    "illegal_rois": total_count - legal_count,
                }
            )
        return jsonify({"error": _ERR_FILE_TYPE}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rois/convert-to-images", methods=["POST"])
def convert_rois_to_images():
    try:
        data = request.json
        rois = data.get("rois", [])
        # min_area = int(data.get('min_area', '10')) # Optional usage

        if not rois:
            return jsonify({"error": "No ROIs provided"}), 400

        if "image_width" not in session_data or "image_height" not in session_data:
            return jsonify({"error": "No input image loaded session"}), 400

        width, height = session_data["image_width"], session_data["image_height"]
        previews = []

        for i, roi in enumerate(rois):
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"])
            is_legal = roi.get("is_legal", True)

            # Create a blank mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Draw the rectangle
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # Save as image
            roi_type = "legal" if is_legal else "illegal"
            timestamp = int(time.time())
            filename = f"roi_{roi_type}_draw_{timestamp}_{i}.png"
            filepath = os.path.join(upload_folder(), filename)
            cv2.imwrite(filepath, mask)

            # Add to session
            if is_legal:
                if "roi_legal_images" not in session_data:
                    session_data["roi_legal_images"] = []
                session_data["roi_legal_images"].append(filepath)
            else:
                if "roi_illegal_images" not in session_data:
                    session_data["roi_illegal_images"] = []
                session_data["roi_illegal_images"].append(filepath)

            # Create preview
            preview_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            if is_legal:
                preview_img[mask == 255] = [255, 255, 255]
            else:
                preview_img[mask == 255] = [255, 0, 0]

            previews.append(
                {
                    "filepath": filepath,
                    "filename": filename,
                    "timestamp": timestamp,
                    "is_legal": is_legal,
                    "roi_count": 1,
                    "image_size": {"width": width, "height": height},
                    "preview": array_to_base64(preview_img),
                }
            )

        return jsonify({"success": True, "total_rois": len(rois), "previews": previews})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/rois/save", methods=["POST"])
def save_rois():
    try:
        data = request.json
        rois = data.get("rois", [])

        if not rois:
            return jsonify({"error": "No ROIs provided"}), 400

        roi_data = {"rois": rois}
        filepath = os.path.join(upload_folder(), "drawn_rois.json")

        with open(filepath, "w") as f:
            json.dump(roi_data, f, indent=2)

        session_data["roi_json"] = filepath
        legal_count = sum(1 for roi in rois if roi.get("is_legal", True))

        return jsonify(
            {
                "success": True,
                "filepath": filepath,
                "total_rois": len(rois),
                "legal_rois": legal_count,
                "illegal_rois": len(rois) - legal_count,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rois/clear-images", methods=["POST"])
def clear_roi_images():
    try:
        data = request.json or {}
        clear_type = data.get("type", "all")  # 'legal', 'illegal', or 'all'

        if clear_type in ["legal", "all"]:
            if "roi_legal_images" in session_data:
                for filepath in session_data["roi_legal_images"]:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                session_data["roi_legal_images"] = []

        if clear_type in ["illegal", "all"]:
            if "roi_illegal_images" in session_data:
                for filepath in session_data["roi_illegal_images"]:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                session_data["roi_illegal_images"] = []

        return jsonify({"success": True, "cleared_type": clear_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rois/delete-image", methods=["POST"])
def delete_roi_image():
    try:
        data = request.json
        filepath = data.get("filepath")
        is_legal = data.get("is_legal")

        if not filepath:
            return jsonify({"error": "Filepath is required"}), 400

        roi_type = "legal" if is_legal else "illegal"
        session_key = f"roi_{roi_type}_images"

        if session_key in session_data and filepath in session_data[session_key]:
            session_data[session_key].remove(filepath)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    log.info(f"Warning: Could not delete file {filepath}: {e}")

            return jsonify({"success": True, "message": f"Deleted {roi_type} ROI image"})
        else:
            return jsonify({"error": "Image not found in session"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rois/clear-json", methods=["POST"])
def clear_roi_json():
    try:
        if "roi_json" in session_data:
            filepath = session_data["roi_json"]
            if os.path.exists(filepath):
                os.remove(filepath)
            del session_data["roi_json"]

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload/roi-image", methods=["POST"])
def upload_roi_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": _ERR_NO_FILE}), 400

        file = request.files["file"]
        is_legal = request.form.get("is_legal", "true").lower() == "true"
        min_area = int(request.form.get("min_area", "10"))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Always append timestamp to ensure uniqueness and history tracking
            roi_type = "legal" if is_legal else "illegal"
            name, ext = os.path.splitext(filename)
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(upload_folder(), f"roi_{roi_type}_{filename}")

            file.save(filepath)

            roi_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if roi_img is None:
                return jsonify({"error": "Could not load image file"}), 400

            if "image_width" in session_data:
                roi_img = cv2.resize(roi_img, (session_data["image_width"], session_data["image_height"]))
                _, roi_img = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite(filepath, roi_img)

            if is_legal:
                if "roi_legal_images" not in session_data:
                    session_data["roi_legal_images"] = []
                session_data["roi_legal_images"].append(filepath)
            else:
                if "roi_illegal_images" not in session_data:
                    session_data["roi_illegal_images"] = []
                session_data["roi_illegal_images"].append(filepath)

            roi_count = 0
            if is_legal and "image_width" in session_data:
                amp_temp = AutoMaskPlacement(session_data["image_width"], session_data["image_height"])
                separated_masks = amp_temp.roi_separator.separate_connected_regions(roi_img, min_area)
                roi_count = len(separated_masks)

            preview_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)
            if is_legal:
                preview_img[roi_img == 255] = [255, 255, 255]
            else:
                preview_img[roi_img == 255] = [255, 0, 0]

            return jsonify(
                {
                    "success": True,
                    "filepath": filepath,
                    "filename": filename,
                    "timestamp": timestamp,
                    "is_legal": is_legal,
                    "roi_count": roi_count,
                    "preview": array_to_base64(preview_img),
                }
            )
        return jsonify({"error": "Invalid file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate_masks():
    try:
        data = request.json
        if "submask" not in session_data:
            return jsonify({"error": "No submask uploaded"}), 400

        has_roi_json = "roi_json" in session_data
        has_roi_images = "roi_legal_images" in session_data and session_data["roi_legal_images"]

        if not has_roi_json and not has_roi_images:
            return jsonify({"error": "No ROIs defined"}), 400

        # Build config
        config = {
            "shift_x_probability": data["shift_x_probability"],
            "shift_y_probability": data["shift_y_probability"],
            "rotation_probability": data["rotation_probability"],
            "scale_x_probability": data["scale_x_probability"],
            "scale_y_probability": data["scale_y_probability"],
            "flip_x_probability": data["flip_x_probability"],
            "flip_y_probability": data["flip_y_probability"],
            "shear_x_probability": data["shear_x_probability"],
            "shear_y_probability": data["shear_y_probability"],
            "morph_probability": data["morph_probability"],
        }

        # Handle augmentations ranges if provided
        for key in [
            "shift_x_range",
            "shift_y_range",
            "rotation_range",
            "scale_x_range",
            "scale_y_range",
            "shear_x_range",
            "shear_y_range",
        ]:
            if key in data and data[key] is not None:
                config[key] = tuple(data[key])

        if "kernel_size" in data:
            config["kernel_size"] = data["kernel_size"]
        if "scale_fixed_ratio" in data:
            config["scale_fixed_ratio"] = data["scale_fixed_ratio"]

        # Specific prob overrides
        for key in [
            "shift_x_probability",
            "shift_y_probability",
            "scale_x_probability",
            "scale_y_probability",
            "shear_x_probability",
            "shear_y_probability",
        ]:
            if key in data and data[key] is not None:
                config[key] = data[key]

        aug_params = AugmentationParams(**config)

        # Dynamic operations settings
        dynamic_ops = {}
        dynamic_ops["use_dynamic_shift_range"] = "shift_x_range" not in config and "shift_y_range" not in config
        dynamic_ops["use_dynamic_rotation_range"] = "rotation_range" not in config
        dynamic_ops["use_dynamic_shear_range"] = "shear_x_range" not in config and "shear_y_range" not in config

        aug_params = replace(aug_params, **dynamic_ops)

        # Handle submask pool mode - randomly select a submask for each generation
        if "submask_pool_files" in session_data and session_data["submask_pool_files"]:
            submask_path = rand_mod.choice(session_data["submask_pool_files"])
            log.info(f"[POOL] Randomly selected submask: {submask_path}")
        else:
            submask_path = session_data["submask"]

        submask_img = Image.open(submask_path)
        original_submask_size = submask_img.size

        # Use input image dimensions if available, otherwise fall back to submask dimensions
        if "image_width" in session_data and "image_height" in session_data:
            width = session_data["image_width"]
            height = session_data["image_height"]
        else:
            width, height = submask_img.size

        # Resize submask to match image dimensions if they differ
        resized_submask_path = submask_path
        if original_submask_size != (width, height):
            submask_img_resized = submask_img.resize((width, height), Image.Resampling.NEAREST)
            resized_submask_path = os.path.join(upload_folder(), "submask_resized.png")
            submask_img_resized.save(resized_submask_path)

        # Parameters
        n_mode = data.get("n_mode", "fixed")
        n_instances_fixed = data.get("n_instances", 1)
        n_range_min = data.get("n_range_min", 1)
        n_range_max = data.get("n_range_max", 1)
        seed_mode = data.get("seed_mode", "single")
        min_area = data.get("min_area", 10)
        max_retry = data.get("max_retry_per_mask", 10)
        separate_rois = data.get("separate_rois", True)

        # Determine seeds
        seeds = []
        if seed_mode == "single":
            s = data.get("seed", 42)
            seeds = [s if s >= 0 else None]
        elif seed_mode == "range":
            seeds = list(range(data.get("seed_start", 1), data.get("seed_end", 10) + 1))
        elif seed_mode == "list":
            seeds = [int(s.strip()) for s in data.get("seed_list", "").split(",") if s.strip()]

        roi_align = AlignmentPoint(data.get("roi_alignment", "random"))
        submask_align_val = data.get("submask_alignment", "None")
        submask_align = AlignmentPoint(submask_align_val) if submask_align_val != "None" else None
        strict_align = data.get("strict_alignment", False)

        all_results = []

        for i, seed in enumerate(seeds):
            output_dir = os.path.join(output_folder(), f"seed_{seed if seed is not None else 'None'}")
            os.makedirs(output_dir, exist_ok=True)

            # Determine n_instances for this seed
            if n_mode == "range":
                if seed is not None:
                    rand_mod.seed(seed)
                n_instances = rand_mod.randint(n_range_min, n_range_max)
            else:
                n_instances = n_instances_fixed

            amp = AutoMaskPlacement(
                width, height, aug_params, roi_align, submask_align, strict_align, seed, max_retry, separate_rois
            )

            # Combine all legal and illegal ROI images from session
            roi_legal_images = session_data.get("roi_legal_images", [])
            roi_illegal_images = session_data.get("roi_illegal_images", [])

            # Add images that might have been uploaded individually if not already in list
            # (Note: Current implementation adds them to lists during upload, so this is just a safety check)

            amp.load_combined_rois(
                json_path=None,  # Explicitly disable JSON path as we use image lists now
                roi_image_paths=roi_legal_images,
                illegal_image_paths=roi_illegal_images,
                min_area=min_area,
            )

            # Save visual for first seed
            if i == 0:
                amp.visualize_rois(os.path.join(output_dir, _ROI_VISUALIZATION_PNG))

            output_paths = amp.process_submask(
                resized_submask_path,
                n_instances,
                output_dir,
                save_cropped_submask=(i == 0),
                save_augmented_masks=True,
                strict_alignment=strict_align,
            )

            if output_paths:
                # output_paths[0] is now a dict with 'output_path', 'n_instances', 'seed'
                output_info = output_paths[0]
                result = {
                    "seed": seed,
                    "output_path": output_info["output_path"],
                    "output_dir": output_dir,
                    "image": image_to_base64(output_info["output_path"]),
                    "n_instances": n_instances,
                }

                # Add intermediate results for first seed
                if i == 0:
                    result["intermediate_results"] = _intermediate_images(output_dir)

                # Add augmented masks
                aug_masks = []
                aug_dir = os.path.join(output_dir, "augmented_masks")
                if os.path.exists(aug_dir):
                    for f in sorted(os.listdir(aug_dir)):
                        if f.endswith(".png"):
                            aug_masks.append({"filename": f, "image": image_to_base64(os.path.join(aug_dir, f))})
                result["augmented_masks"] = aug_masks
                all_results.append(result)

        if not all_results:
            return jsonify({"error": "Failed to generate masks"}), 500

        # Save metadata for session restoration
        session_data["results_metadata"] = []
        for res in all_results:
            session_data["results_metadata"].append(
                {
                    "seed": res["seed"],
                    "output_path": res["output_path"],
                    "output_dir": res["output_dir"],
                    "n_instances": res["n_instances"],
                }
            )

        session_data["generated_mask"] = all_results[0]["output_path"]

        # --- Generate Heatmap (if possible) ---
        heatmap_data = None
        try:
            # Only try to generate heatmap if we have results
            if all_results:
                # Use default params for heatmap
                hm_result, hm_error = overlay_heatmap_impl(
                    output_folder(), overlay_on_input=True, opacity=0.5, apply_colormap=True
                )
                if not hm_error:
                    heatmap_data = hm_result
        except Exception as e:
            log.info(f"Error auto-generating heatmap: {e}")
        # --------------------------------------

        return jsonify(
            {
                "success": True,
                "results": all_results,
                "total_seeds": len(all_results),
                "image": all_results[0]["image"],
                "intermediate_results": all_results[0].get("intermediate_results", {}),
                "heatmap": heatmap_data,
            }
        )

    except Exception as e:
        # Printed, not returned: the traceback names server paths and internals, and the caller can
        # do nothing with them. Same shape as every other handler here.
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/render", methods=["POST"])
def render_mask():
    try:
        data = request.json
        if "input_image" not in session_data:
            return jsonify({"error": "No input image uploaded"}), 400

        mask_path = None
        if data.get("seed") is not None:
            seed_str = str(data.get("seed")) if data.get("seed") != "None" else "None"
            path = os.path.join(output_folder(), f"seed_{seed_str}")
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.startswith("auto_placed_mask") and f.endswith(".png"):
                        mask_path = os.path.join(path, f)
                        break

        if not mask_path:
            mask_path = session_data.get("generated_mask")

        if not mask_path or not os.path.exists(mask_path):
            return jsonify({"error": "Mask not found"}), 400

        opacity = data.get("opacity", 60) / 100.0
        color = data.get("color", [255, 0, 0])

        base = Image.open(session_data["input_image"]).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if base.size != mask.size:
            mask = mask.resize(base.size, Image.NEAREST)

        base_arr = np.array(base).astype(np.float32)
        mask_arr = np.array(mask).astype(np.float32) / 255.0

        overlay = np.zeros_like(base_arr)
        for i in range(3):
            overlay[:, :, i] = mask_arr * color[i]

        mask_alpha = mask_arr[:, :, np.newaxis] * opacity
        result = base_arr * (1 - mask_alpha) + overlay * opacity
        result = np.clip(result, 0, 255).astype(np.uint8)

        return jsonify({"success": True, "image": array_to_base64(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def overlay_heatmap_impl(directory, overlay_on_input=False, opacity=0.5, apply_colormap=True):
    """
    Implementation of overlay_heatmap logic, separated from the route handler.
    """
    try:
        # Find all auto_placed_mask images recursively
        mask_files = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.startswith("auto_placed_mask") and filename.endswith(".png"):
                    mask_files.append(os.path.join(root, filename))

        if len(mask_files) == 0:
            return None, "No auto_placed_mask images found in directory"

        log.info(f"Found {len(mask_files)} mask images to overlay")

        # Initialize accumulator
        first_mask = Image.open(mask_files[0]).convert("L")
        height, width = first_mask.size[1], first_mask.size[0]
        accumulator = np.zeros((height, width), dtype=np.float32)

        # Accumulate all masks
        for i, mask_path in enumerate(mask_files):
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask, dtype=np.float32) / 255.0
            accumulator += mask_array

        # Normalize to 0-255 range
        if accumulator.max() > 0:
            heatmap = (accumulator / accumulator.max() * 255).astype(np.uint8)
        else:
            heatmap = accumulator.astype(np.uint8)

        # Apply colormap for better visualization
        if apply_colormap:
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        else:
            heatmap_colored = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)

        # Optional: Overlay on input image if available
        if overlay_on_input and "input_image" in session_data:
            base_img = Image.open(session_data["input_image"]).convert("RGB")

            if base_img.size != (width, height):
                heatmap_colored_resized = cv2.resize(heatmap_colored, base_img.size)
            else:
                heatmap_colored_resized = heatmap_colored

            base_array = np.array(base_img).astype(np.float32)
            result = base_array * (1 - opacity) + heatmap_colored_resized.astype(np.float32) * opacity
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = heatmap_colored

        # Save the heatmap
        output_filename = f"heatmap_overlay_{len(mask_files)}_masks.png"
        output_path = os.path.join(output_folder(), output_filename)
        Image.fromarray(result).save(output_path)

        log.info(f"Heatmap saved to: {output_path}")

        return (
            {
                "success": True,
                "image": array_to_base64(result),
                "num_masks": len(mask_files),
                "max_overlap": int(accumulator.max()),
                "output_path": output_path,
                "filename": output_filename,
            },
            None,
        )

    except Exception as e:
        traceback.print_exc()
        return None, str(e)


@app.route("/api/overlay-heatmap", methods=["POST"])
def overlay_heatmap():
    """
    Generate a heatmap by overlaying all auto_placed_mask images from a directory.
    Areas that appear more frequently will be darker/more intense.
    """
    try:
        data = request.json
        directory = data.get("directory", None)

        # Third endpoint taking a directory from the request, so the same boundary as the output
        # directory and the submask pool. Narrower than those two — it only reads a fixed filename
        # prefix and writes nothing — but it os.walks whatever it is given and returns the masks it
        # finds composited into the response as base64, so without a root it reads any
        # auto_placed_mask*.png on the host and hands it back to the caller.
        if directory:
            try:
                directory = _resolve_output_directory(directory)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        else:
            directory = output_folder()

        if not os.path.exists(directory):
            return jsonify({"error": f"Directory not found: {directory}"}), 400

        opacity = data.get("opacity", 50) / 100.0
        apply_colormap = data.get("colormap", True)
        overlay_on_input = data.get("overlayOnInput", False)

        result, error = overlay_heatmap_impl(directory, overlay_on_input, opacity, apply_colormap)

        if error:
            return jsonify({"error": error}), 400 if "not found" in error else 500

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/results", methods=["GET"])
def download_results():
    """
    Download the entire output folder as a zip file.
    """

    try:
        output_dir = output_folder()

        if not os.path.exists(output_dir):
            return jsonify({"error": "Output directory not found"}), 400

        # Check if there are any files to download
        file_count = sum(len(files) for _, _, files in os.walk(output_dir))
        if file_count == 0:
            return jsonify({"error": "No files to download. Generate masks first."}), 400

        # Create a timestamp for the zip filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"amp_results_{timestamp}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)

        # Create zip file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        log.info(f"Created zip file: {zip_path} with {file_count} files")

        return send_file(zip_path, mimetype="application/zip", as_attachment=True, download_name=zip_filename)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/results-info", methods=["GET"])
def get_results_info():
    """
    Get information about the current results folder.
    """
    try:
        output_dir = output_folder()

        if not os.path.exists(output_dir):
            return jsonify({"exists": False, "file_count": 0, "folder_count": 0, "total_size": 0, "path": output_dir})

        file_count = 0
        folder_count = 0
        total_size = 0
        seed_folders = []

        for root, dirs, files in os.walk(output_dir):
            folder_count += len(dirs)
            file_count += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)

            # Count seed folders at top level
            if root == output_dir:
                seed_folders = [d for d in dirs if d.startswith("seed_")]

        return jsonify(
            {
                "exists": True,
                "file_count": file_count,
                "folder_count": folder_count,
                "seed_count": len(seed_folders),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "path": output_dir,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/clear", methods=["POST"])
def clear_results():
    """
    Clear all generated results (folders starting with seed_) from the output directory.
    """
    try:
        output_dir = output_folder()

        if not os.path.exists(output_dir):
            return jsonify({"error": "Output directory not found"}), 400

        cleared_count = 0

        for item in os.listdir(output_dir):
            if item.startswith("seed_"):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    cleared_count += 1

        # Also clear session data related to results
        if "results_metadata" in session_data:
            session_data["results_metadata"] = []
        if "generated_mask" in session_data:
            del session_data["generated_mask"]
        if "heatmap" in session_data:
            del session_data["heatmap"]

        # Also try to clear heatmap images
        for item in os.listdir(output_dir):
            if item.startswith("heatmap_overlay_") and item.endswith(".png"):
                try:
                    os.remove(os.path.join(output_dir, item))
                except Exception:
                    pass

        return jsonify(
            {"success": True, "message": f"Cleared {cleared_count} seed result folders", "cleared_count": cleared_count}
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --- ROI Generation Endpoints ---


@app.route("/api/roi/config/default", methods=["GET"])
def get_default_roi_config():
    """Get default filter configuration for ROI generation"""
    # Return default config structure from helper
    return jsonify(get_default_roi_generation_config())


def run_roi_generation(image_path, boxes, output_dir, config_overrides):
    """Run ROI generation logic"""
    if not cached_roi_models:
        return False, "Models not initialized"

    try:
        # Prepare sample
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {"image_path": image_path, "image": image, "boxes": boxes}

        # Build overrides for OmegaConf
        overrides = []
        overrides.append("save_visualization=false")

        if config_overrides:
            c = config_overrides

            # grayscale_to_mask
            if "grayscale_to_mask_enabled" in c:
                overrides.append(f"grayscale_to_mask.enabled={str(c['grayscale_to_mask_enabled']).lower()}")
            if "grayscale_to_mask_threshold_mode" in c:
                overrides.append(f"grayscale_to_mask.threshold_mode={c['grayscale_to_mask_threshold_mode']}")
            if "grayscale_to_mask_threshold_value" in c:
                overrides.append(f"grayscale_to_mask.threshold_value={int(c['grayscale_to_mask_threshold_value'])}")

            # box_to_mask
            if "box_to_mask_enabled" in c:
                overrides.append(f"box_to_mask.enabled={str(c['box_to_mask_enabled']).lower()}")

            # template_box_to_masks
            if "template_box_to_masks_enabled" in c:
                overrides.append(f"template_box_to_masks.enabled={str(c['template_box_to_masks_enabled']).lower()}")
            if "max_template_boxes" in c:
                overrides.append(f"template_box_to_masks.max_template={int(c['max_template_boxes'])}")
            if "max_proposal" in c:
                overrides.append(f"template_box_to_masks.max_proposal={int(c['max_proposal'])}")
            if "proposal_similarity_tol" in c:
                overrides.append(f"template_box_to_masks.proposal_similarity_tol={float(c['proposal_similarity_tol'])}")
            if "box_enabled" in c:
                overrides.append(f"template_box_to_masks.box_enabled={str(c['box_enabled']).lower()}")
            if "size_tol" in c:
                overrides.append(f"template_box_to_masks.size_tol={float(c['size_tol'])}")
            if "aspect_tol" in c:
                overrides.append(f"template_box_to_masks.aspect_tol={float(c['aspect_tol'])}")
            if "mask_enabled" in c:
                overrides.append(f"template_box_to_masks.mask_enabled={str(c['mask_enabled']).lower()}")
            if "chamfer_tol" in c:
                overrides.append(f"template_box_to_masks.chamfer_tol={float(c['chamfer_tol'])}")
            if "hog_enabled" in c:
                overrides.append(f"template_box_to_masks.hog_enabled={str(c['hog_enabled']).lower()}")
            if "hog_similarity_tol" in c:
                overrides.append(f"template_box_to_masks.hog_similarity_tol={float(c['hog_similarity_tol'])}")
            if "color_hist_enabled" in c:
                overrides.append(f"template_box_to_masks.color_hist_enabled={str(c['color_hist_enabled']).lower()}")
            if "lightness_tol" in c:
                overrides.append(f"template_box_to_masks.lightness_tol={float(c['lightness_tol'])}")
            if "color_tol" in c:
                overrides.append(f"template_box_to_masks.color_tol={float(c['color_tol'])}")

            # --- Post Process ---
            if "morphological_operation" in c:
                val = c["morphological_operation"]
                if val and val != "null" and val != "None":
                    overrides.append(f"morphological_operation={val}")
                else:
                    overrides.append("morphological_operation=null")

            if "morphological_kernel" in c:
                # Expecting list [k, k]
                k = c["morphological_kernel"]
                if isinstance(k, list) and len(k) == 2:
                    overrides.append(f"morphological_kernel=[{k[0]},{k[1]}]")

            if "refine_template" in c:
                overrides.append(f"template_box_to_masks.refine_template={str(c['refine_template']).lower()}")

            if "rotation_degrees" in c:
                degrees = c["rotation_degrees"]
                if isinstance(degrees, list):
                    deg_str = ",".join(map(str, degrees))
                    overrides.append(f"template_box_to_masks.rotation_degrees=[{deg_str}]")

            if "allow_flip" in c:
                flips = c["allow_flip"]
                if isinstance(flips, list):
                    flip_str = ",".join(flips)
                    overrides.append(f"template_box_to_masks.allow_flip=[{flip_str}]")
            # --------------------

        # Load and merge config
        default_cfg = OmegaConf.structured(ROIGenerationConfig)
        cli_cfg = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(default_cfg, cli_cfg)

        sample["config"] = dict(config)

        # Run pipeline
        run_roi_pipeline([sample], output_dir, cached_roi_models)

        return True, None
    except Exception as e:
        traceback.print_exc()
        return False, str(e)


@app.route("/api/roi/generate", methods=["POST"])
def generate_roi():
    """Generate ROI masks from bounding boxes"""
    if not cached_roi_models:
        return jsonify({"error": "ROI models are initializing"}), 503

    try:
        # Use existing input image if available
        if "input_image" not in session_data:
            if "image" not in request.files:
                return jsonify({"error": "No input image (upload first or provide in request)"}), 400

            # Handle uploaded image
            f = request.files["image"]
            filename = secure_filename(f.filename)
            filepath = os.path.join(upload_folder(), f"roi_input_{filename}")
            f.save(filepath)
            session_data["ori_filename"] = filename
            session_data["input_image"] = filepath

        image_path = session_data["input_image"]

        # Get bboxes
        bboxes_json = request.form.get("bboxes", "[]")
        bboxes = json.loads(bboxes_json)

        # Get config
        config_json = request.form.get("config", "{}")
        user_config = json.loads(config_json)

        # Prepare format [x0, y0, x1, y1]
        boxes = []
        for b in bboxes:
            if isinstance(b, dict) and "x" in b:
                boxes.append([float(b["x"]), float(b["y"]), float(b["x"] + b["width"]), float(b["y"] + b["height"])])
            elif isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append([float(v) for v in b])

        # Create output dir
        output_dir = os.path.join(output_folder(), f"roi_gen_{session_data['ori_filename']}")
        os.makedirs(output_dir, exist_ok=True)

        # Run generation
        success, error = run_roi_generation(image_path, boxes, output_dir, user_config)

        if not success:
            return jsonify({"error": f"Generation failed: {error}"}), 500

        # Process results from all methods
        sample_dir = os.path.join(output_dir, "sample_00001")

        methods_map = {
            "grayscale_to_mask": "Bright or dark areas\n(Grayscale-to-Mask)",
            "box_to_mask": "Specific objects\n(Box-to-Mask)",
            "template_box_to_masks": "Many similar objects\n(Template-Box-to-Masks)",
        }

        generated_results = []
        orig_img = Image.open(image_path).convert("RGB")
        w, h = orig_img.size

        primary_roi_gen_url = None
        stage_results = {}

        # Helper to extract masks from binary mask
        def extract_masks_from_binary(bin_mask_path, target_size):
            if not os.path.exists(bin_mask_path):
                return 0, None

            bin_mask = cv2.imread(bin_mask_path, cv2.IMREAD_GRAYSCALE)
            target_w, target_h = target_size

            if bin_mask.shape[:2] != (target_h, target_w):
                bin_mask = cv2.resize(bin_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            num_labels, _ = cv2.connectedComponents(bin_mask)

            return num_labels - 1, binary_mask_to_base64(bin_mask)

        # Iterate through methods
        for method_key, method_name in methods_map.items():
            out_dir = os.path.join(sample_dir, method_key, "output")
            bin_mask_path = os.path.join(out_dir, "binary_mask.png")

            if os.path.exists(bin_mask_path):
                count, method_b64 = extract_masks_from_binary(bin_mask_path, (w, h))

                result_entry = {
                    "method": method_key,
                    "name": method_name,
                    "roiGeneratedUrl": method_b64,
                    "count": count,
                }
                generated_results.append(result_entry)

                # Use Template Matching as primary, or first available
                if method_key == "template_box_to_masks" or not primary_roi_gen_url:
                    primary_roi_gen_url = method_b64

                    # Process result.json to get detailed stage results and images
                    result_json_path = os.path.join(out_dir, "result.json")
                    if os.path.exists(result_json_path):
                        try:
                            # Load binary mask for this method
                            bin_mask = cv2.imread(bin_mask_path, cv2.IMREAD_GRAYSCALE)
                            if bin_mask.shape[:2] != (h, w):
                                bin_mask = cv2.resize(bin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                            full_config = merge_config_with_defaults(user_config)

                            results_tb2m = process_stage_result_of_template_box_to_masks(
                                result_json_path=result_json_path,
                                config=full_config,
                                image=orig_img,
                                binary_mask=bin_mask,
                                bboxes=bboxes,
                            )

                            if "stageResults" in results_tb2m:
                                stage_results = results_tb2m["stageResults"]
                            if "stageImages" in results_tb2m:
                                session_data["stage_images"] = results_tb2m["stageImages"]  # Store if needed later
                                stage_images = results_tb2m["stageImages"]
                        except Exception as e:
                            log.info(f"Error processing stage results: {e}")
                            traceback.print_exc()

        if not generated_results:
            return jsonify({"error": "No masks generated from any method"}), 500

        return jsonify(
            {
                "success": True,
                "stageResults": stage_results,
                "stageImages": stage_images if "stage_images" in locals() else {},
                "generated_results": generated_results,  # New list with all results
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("🚀 Automatic Mask Placement + ROI Generation GUI")
    log.info("=" * 60)
    log.info(f"📁 Upload: {_UPLOAD_BASE}")
    log.info(f"💾 Output: {_OUTPUT_BASE}")

    # Start ROI init thread
    threading.Thread(target=background_init, daemon=True).start()

    port = int(os.environ.get("AMP_PORT", 5000))
    log.info(f"🌐 Open http://127.0.0.1:{port}/?token={_AUTH_TOKEN}")
    # debug=True exposes the Werkzeug interactive debugger (remote code execution
    # on any unhandled exception); bind to loopback and keep the debugger off.
    app.run(debug=False, host="127.0.0.1", port=port)
