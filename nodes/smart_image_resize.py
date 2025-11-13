import torch
import numpy as np
from PIL import Image, ImageOps
import comfy.utils
from comfy import model_management
from .ImgMods import tensor2pil, pil2tensor

MAX_RESOLUTION = 8192

# Standard aspect ratios: 3:4, 4:3, 1:1, 16:9, 9:16
STANDARD_RATIOS = [
    (3, 4),  # Portrait 3:4
    (4, 3),  # Landscape 4:3
    (1, 1),  # Square 1:1
    (16, 9),  # Landscape 16:9
    (9, 16),  # Portrait 9:16
]


class SmartImageResize:
    """
    Smart image resize node that adjusts images based on min/max resolution constraints.
    Supports both model-based upscaling (if upscale_model provided) and basic interpolation
    upscaling (if no model). Can add padding to fit standard aspect ratios.
    """

    upscale_methods = ["area", "lanczos", "bilinear", "nearest-exact", "bicubic"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_resolution": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "max_resolution": (
                    "INT",
                    {"default": 1536, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "upscale_factor": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.1, "max": 4.0, "step": 0.1},
                ),
                "upscale_method": (cls.upscale_methods,),
                "standard_ratio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_reference": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "smart_resize"
    CATEGORY = "Image Resizing"

    def find_closest_standard_ratio(self, width, height):
        """Find the closest standard aspect ratio to the given dimensions."""
        current_ratio = width / height if height > 0 else 1.0
        min_diff = float("inf")
        best_ratio = STANDARD_RATIOS[0]

        for ratio_w, ratio_h in STANDARD_RATIOS:
            ratio_value = ratio_w / ratio_h
            diff = abs(current_ratio - ratio_value)
            if diff < min_diff:
                min_diff = diff
                best_ratio = (ratio_w, ratio_h)

        return best_ratio

    def add_padding_to_ratio(
        self, image, target_ratio_w, target_ratio_h, min_res, max_res
    ):
        """Add black padding to fit the target aspect ratio within min/max constraints."""
        pil_img = tensor2pil(image)
        current_w, current_h = pil_img.size
        target_ratio = target_ratio_w / target_ratio_h

        # First, if current dimensions exceed max_res, scale down to fit max_res
        # while maintaining aspect ratio
        if max(current_w, current_h) > max_res:
            longest = max(current_w, current_h)
            scale_factor = max_res / longest
            new_current_w = int(current_w * scale_factor)
            new_current_h = int(current_h * scale_factor)

            # Scale the image
            pil_img = pil_img.resize(
                (new_current_w, new_current_h), Image.Resampling.LANCZOS
            )
            current_w = new_current_w
            current_h = new_current_h

        # Now calculate target dimensions to fit the target ratio within constraints
        # Try to fit based on the longer dimension first
        if current_w >= current_h:
            # Image is landscape or square - try to fit based on width
            candidate_w = min(max(current_w, min_res), max_res)
            candidate_h = int(candidate_w / target_ratio)

            # If height is out of bounds, recalculate based on height instead
            if candidate_h < min_res:
                candidate_h = min_res
                candidate_w = int(candidate_h * target_ratio)
                if candidate_w > max_res:
                    candidate_w = max_res
                    candidate_h = int(candidate_w / target_ratio)
            elif candidate_h > max_res:
                candidate_h = max_res
                candidate_w = int(candidate_h * target_ratio)
                if candidate_w < min_res:
                    candidate_w = min_res
                    candidate_h = int(candidate_w / target_ratio)
        else:
            # Image is portrait - try to fit based on height
            candidate_h = min(max(current_h, min_res), max_res)
            candidate_w = int(candidate_h * target_ratio)

            # If width is out of bounds, recalculate based on width instead
            if candidate_w < min_res:
                candidate_w = min_res
                candidate_h = int(candidate_w / target_ratio)
                if candidate_h > max_res:
                    candidate_h = max_res
                    candidate_w = int(candidate_h * target_ratio)
            elif candidate_w > max_res:
                candidate_w = max_res
                candidate_h = int(candidate_w / target_ratio)
                if candidate_h < min_res:
                    candidate_h = min_res
                    candidate_w = int(candidate_h * target_ratio)

        # Final bounds check
        new_width = max(min_res, min(candidate_w, max_res))
        new_height = max(min_res, min(candidate_h, max_res))

        # Ensure exact ratio is maintained
        actual_ratio = new_width / new_height if new_height > 0 else 1.0
        if actual_ratio > target_ratio:
            # Width is too large for the ratio
            new_width = int(new_height * target_ratio)
        else:
            # Height is too large for the ratio
            new_height = int(new_width / target_ratio)

        # Ensure minimum dimensions are respected
        if new_width < min_res or new_height < min_res:
            new_width = max(min_res, new_width)
            new_height = max(min_res, new_height)
            new_height = int(new_width / target_ratio)

        # Calculate padding needed
        pad_w = max(0, new_width - current_w)
        pad_h = max(0, new_height - current_h)

        # Distribute padding evenly
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Add black padding
        padded_img = ImageOps.expand(
            pil_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0)
        )

        return pil2tensor(padded_img), new_width, new_height

    def match_reference_ratio(self, image, reference_image):
        """Add padding to match the reference image's aspect ratio."""
        pil_img = tensor2pil(image)
        ref_pil = tensor2pil(reference_image)

        current_w, current_h = pil_img.size
        ref_w, ref_h = ref_pil.size
        target_ratio = ref_w / ref_h if ref_h > 0 else 1.0
        current_ratio = current_w / current_h if current_h > 0 else 1.0

        # Calculate target dimensions to match reference ratio
        if current_ratio < target_ratio:
            # Image is narrower than target - pad width
            new_width = int(current_h * target_ratio)
            new_height = current_h
        else:
            # Image is wider than target - pad height
            new_width = current_w
            new_height = int(current_w / target_ratio)

        # Calculate padding needed
        pad_w = max(0, new_width - current_w)
        pad_h = max(0, new_height - current_h)

        # Distribute padding evenly
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Add black padding
        padded_img = ImageOps.expand(
            pil_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0)
        )

        return pil2tensor(padded_img), new_width, new_height

    def upscale_basic(self, image, factor, upscale_method):
        """Upscale image using basic interpolation method (no model)."""
        old_width = image.shape[2]
        old_height = image.shape[1]
        new_width = int(old_width * factor)
        new_height = int(old_height * factor)

        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(
            samples, new_width, new_height, upscale_method, crop="disabled"
        )
        s = s.movedim(1, -1)

        return s

    def upscale_with_model(self, image, upscale_model, factor, upscale_method):
        """Upscale image using the upscale model."""
        device = model_management.get_torch_device()
        upscale_model.to(device)

        # First upscale with model
        in_img = image.movedim(-1, -3).to(device)
        s = comfy.utils.tiled_scale(
            in_img,
            lambda a: upscale_model(a),
            tile_x=128 + 64,
            tile_y=128 + 64,
            overlap=8,
            upscale_amount=upscale_model.scale,
        )
        upscale_model.cpu()
        upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

        # Get dimensions and apply additional scaling if needed
        old_width = image.shape[2]
        old_height = image.shape[1]
        new_width = int(old_width * factor)
        new_height = int(old_height * factor)

        # Apply final scaling
        samples = upscaled.movedim(-1, 1)
        s = comfy.utils.common_upscale(
            samples, new_width, new_height, upscale_method, crop="disabled"
        )
        s = s.movedim(1, -1)

        return s

    def smart_resize(
        self,
        image,
        min_resolution,
        max_resolution,
        upscale_factor,
        standard_ratio,
        upscale_method,
        upscale_model=None,
        image_reference=None,
    ):
        """
        Main resize logic:
        1. If image_reference provided, match its aspect ratio first (using padding)
        2. Calculate sum = min_resolution + max_resolution
        3. Check image dimensions and apply appropriate resizing/upscaling
        4. If standard_ratio is True, add padding to fit standard ratios

        Args:
            upscale_model: Optional upscale model. If provided, uses model-based upscaling.
                          If None, uses basic interpolation method (upscale_method).
            image_reference: Optional reference image. If provided, input image will be padded
                            to match the reference image's aspect ratio.
        """
        # If reference image is provided, match its aspect ratio first
        if image_reference is not None:
            # Process each image in the batch with the reference
            batch_size = image.shape[0]
            ref_batch_size = image_reference.shape[0]
            processed_images = []
            ref_width = 0
            ref_height = 0

            for i in range(batch_size):
                img = image[i : i + 1]
                # Use corresponding reference image or first one if batch sizes don't match
                ref_idx = min(i, ref_batch_size - 1)
                ref_img = image_reference[ref_idx : ref_idx + 1]

                # Match reference ratio using padding
                img, ref_width, ref_height = self.match_reference_ratio(img, ref_img)
                processed_images.append(img)

            # Combine batch
            image = (
                torch.cat(processed_images, dim=0)
                if len(processed_images) > 1
                else processed_images[0]
            )
            print(f"Matched reference ratio: {ref_width}x{ref_height}")

        # Get current image dimensions
        batch_size = image.shape[0]
        current_height = image.shape[1]
        current_width = image.shape[2]

        # Calculate thresholds for scenario 3
        resolution_sum = min_resolution + max_resolution
        # Area-based threshold: use min * max as a reasonable area target
        # This is better for extreme aspect ratios than sum alone
        area_threshold = min_resolution * max_resolution
        # Maximum allowed longest side (allow some tolerance, e.g., 1.5x max)
        max_longest_side = int(max_resolution * 1.5)

        print(f"Input image: {current_width}x{current_height}")
        print(f"Min resolution: {min_resolution}, Max resolution: {max_resolution}")
        print(f"Resolution sum: {resolution_sum}, Area threshold: {area_threshold}")
        print(f"Max longest side tolerance: {max_longest_side}")

        # Process each image in the batch
        processed_images = []
        final_width = 0
        final_height = 0

        for i in range(batch_size):
            img = image[i : i + 1]
            width = current_width
            height = current_height

            # Scenario 1: Image is too large (longest side > max_resolution)
            longest_side = max(width, height)
            if longest_side > max_resolution:
                # Resize to fit max_resolution while keeping aspect ratio
                scale_factor = max_resolution / longest_side
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                print(
                    f"Scenario 1: Resizing from {width}x{height} to {new_width}x{new_height}"
                )

                # Use simple resize (downscale)
                samples = img.movedim(-1, 1)
                resized = comfy.utils.common_upscale(
                    samples, new_width, new_height, upscale_method, crop="disabled"
                )
                img = resized.movedim(1, -1)
                width = new_width
                height = new_height

            # Scenario 2 & 3: Image is too small or needs upscaling
            shortest_side = min(width, height)
            longest_side = max(width, height)

            # Check if we need to upscale
            needs_upscale = shortest_side < min_resolution
            scenario3_passed = False

            if needs_upscale:
                # Keep upscaling until we meet minimum requirements
                iteration = 0
                max_iterations = 10  # Safety limit

                while shortest_side < min_resolution and iteration < max_iterations:
                    iteration += 1
                    print(f"Upscale iteration {iteration}: {width}x{height}")

                    # Predict dimensions after upscaling (before actually upscaling)
                    predicted_width = int(width * upscale_factor)
                    predicted_height = int(height * upscale_factor)
                    predicted_shortest = min(predicted_width, predicted_height)
                    predicted_longest = max(predicted_width, predicted_height)

                    # Check if predicted dimensions would exceed tolerance
                    would_exceed_tolerance = predicted_longest > max_longest_side

                    # If upscaling would exceed tolerance, check scenario 3 with current dimensions first
                    if would_exceed_tolerance:
                        # Check scenario 3 conditions with current dimensions before upscaling
                        if (
                            longest_side > max_resolution
                            and shortest_side < min_resolution
                        ):
                            current_area = width * height
                            current_sum = width + height

                            # Check 1: Area threshold (more meaningful for extreme ratios)
                            area_check = current_area >= area_threshold

                            # Check 2: Sum threshold (for normal aspect ratios)
                            sum_check = current_sum >= resolution_sum

                            # Check 3: Longest side doesn't exceed reasonable tolerance
                            dimension_check = longest_side <= max_longest_side

                            # Pass if area OR sum check passes, AND dimension is within tolerance
                            if (area_check or sum_check) and dimension_check:
                                print(
                                    f"Scenario 3: Checks passed (before upscale) - "
                                    f"Area: {current_area} >= {area_threshold} ({area_check}), "
                                    f"Sum: {current_sum} >= {resolution_sum} ({sum_check}), "
                                    f"Longest: {longest_side} <= {max_longest_side} ({dimension_check})"
                                )
                                scenario3_passed = True
                                break
                            else:
                                # Would exceed tolerance and scenario 3 not met, stop upscaling
                                print(
                                    f"Scenario 3: Stopping before upscale - predicted longest side "
                                    f"{predicted_longest} would exceed tolerance {max_longest_side}, "
                                    f"and current dimensions don't meet scenario 3 criteria"
                                )
                                break
                        else:
                            # Would exceed tolerance but not in scenario 3 condition, stop
                            print(
                                f"Stopping before upscale - predicted longest side {predicted_longest} "
                                f"would exceed tolerance {max_longest_side}"
                            )
                            break

                    # Safe to upscale - proceed with upscaling
                    if upscale_model is not None:
                        # Use model-based upscaling
                        img = self.upscale_with_model(
                            img, upscale_model, upscale_factor, upscale_method
                        )
                    else:
                        # Use basic upscaling method
                        img = self.upscale_basic(img, upscale_factor, upscale_method)

                    # Update dimensions
                    width = predicted_width
                    height = predicted_height
                    shortest_side = predicted_shortest
                    longest_side = predicted_longest

                    # Scenario 3: Check if one side exceeds max but constraints are acceptable
                    # Use area-based check (better for extreme aspect ratios) combined with
                    # max dimension constraint to prevent excessive upscaling
                    if longest_side > max_resolution and shortest_side < min_resolution:
                        current_area = width * height
                        current_sum = width + height

                        # Check 1: Area threshold (more meaningful for extreme ratios)
                        area_check = current_area >= area_threshold

                        # Check 2: Sum threshold (for normal aspect ratios)
                        sum_check = current_sum >= resolution_sum

                        # Check 3: Longest side doesn't exceed reasonable tolerance
                        # This prevents extreme dimensions even if area/sum checks pass
                        dimension_check = longest_side <= max_longest_side

                        # Pass if area OR sum check passes, AND dimension is within tolerance
                        if (area_check or sum_check) and dimension_check:
                            print(
                                f"Scenario 3: Checks passed - "
                                f"Area: {current_area} >= {area_threshold} ({area_check}), "
                                f"Sum: {current_sum} >= {resolution_sum} ({sum_check}), "
                                f"Longest: {longest_side} <= {max_longest_side} ({dimension_check})"
                            )
                            scenario3_passed = True
                            break

                # Final check: if longest side exceeds max, resize down (unless scenario 3 passed)
                if longest_side > max_resolution and not scenario3_passed:
                    scale_factor = max_resolution / longest_side
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)

                    print(f"Final resize: {width}x{height} to {new_width}x{new_height}")

                    samples = img.movedim(-1, 1)
                    resized = comfy.utils.common_upscale(
                        samples, new_width, new_height, upscale_method, crop="disabled"
                    )
                    img = resized.movedim(1, -1)
                    width = new_width
                    height = new_height

            # Apply standard ratio padding if enabled
            if standard_ratio:
                ratio_w, ratio_h = self.find_closest_standard_ratio(width, height)
                print(f"Standard ratio: Applying {ratio_w}:{ratio_h} ratio")

                img, width, height = self.add_padding_to_ratio(
                    img, ratio_w, ratio_h, min_resolution, max_resolution
                )
                print(f"After padding: {width}x{height}")

            processed_images.append(img)
            final_width = width
            final_height = height

        # Combine batch
        result = (
            torch.cat(processed_images, dim=0)
            if len(processed_images) > 1
            else processed_images[0]
        )

        print(f"Final output: {final_width}x{final_height}")

        return (result, final_width, final_height)
