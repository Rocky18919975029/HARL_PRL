from PIL import Image, ImageDraw, ImageFont, ImageChops
import os
import math

# ===== input paths =====
img_paths = [
    "ablation/grf_3_vs_1_with_keeper/eval_score_rate.png",
    "ablation/grf_counterattack_hard/eval_score_rate.png",
    "ablation/mamujoco_humanoid_17x1/eval_average_episode_rewards.png",
    # "ablation/dexhands_ShadowHandOver/train_episode_rewards.png",
]

# ===== layout config =====
layout = "1x3"   # options: "4x1", "2x2", "1x4", "1x2", "2x1", etc.

# 更细粒度控制横向/纵向留白
margin_x = 16
margin_y = 12
gap_x = 12
gap_y = 12

# 是否自动裁掉每张图外围白边
trim_whitespace = True
trim_pad = 4

# 统一缩放方式：推荐按高度统一，避免变形
unify_by = "height"   # "height" or "width"

# 是否允许放大到统一尺寸
allow_upscale = False

add_labels = True
label_style = "plain"   # "plain" or "box"
label_font_size = 26
label_pad_x = 10
label_pad_y = 8

out_png = "ablation/ablation_combined_1x3.png"
out_pdf = "ablation/ablation_combined_1x3.pdf"


def parse_layout(layout: str):
    """
    Parse layout string like '2x2', '4x1', '1x4'.
    Return rows, cols.
    """
    rows, cols = layout.lower().split("x")
    return int(rows), int(cols)


def load_font(font_size: int):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]

    for fp in font_candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, font_size)

    return ImageFont.load_default()


def crop_white_border(img: Image.Image, pad: int = 4, bg_threshold: int = 245):
    """
    Crop near-white outer border.
    Works well for matplotlib-style figures with white background.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Create mask of non-white pixels
    gray = img.convert("L")
    # pixels darker than threshold are treated as content
    mask = gray.point(lambda p: 255 if p < bg_threshold else 0)

    bbox = mask.getbbox()
    if bbox is None:
        return img

    left, upper, right, lower = bbox
    left = max(0, left - pad)
    upper = max(0, upper - pad)
    right = min(img.width, right + pad)
    lower = min(img.height, lower + pad)

    return img.crop((left, upper, right, lower))


def resize_preserve_aspect(img: Image.Image, target_value: int, mode: str, allow_upscale=False):
    """
    Resize while preserving aspect ratio.
    mode = 'height' or 'width'
    """
    w, h = img.size

    if mode == "height":
        if (not allow_upscale) and h <= target_value:
            return img
        scale = target_value / h
    elif mode == "width":
        if (not allow_upscale) and w <= target_value:
            return img
        scale = target_value / w
    else:
        raise ValueError("mode must be 'height' or 'width'")

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def compute_target_size(imgs, unify_by="height"):
    """
    Use the minimum size as the shared reference, to avoid unintended enlargement.
    """
    if unify_by == "height":
        return min(img.height for img in imgs)
    elif unify_by == "width":
        return min(img.width for img in imgs)
    else:
        raise ValueError("unify_by must be 'height' or 'width'")


def combine_images(
    img_paths,
    layout="2x2",
    margin_x=16,
    margin_y=12,
    gap_x=12,
    gap_y=12,
    trim_whitespace=True,
    trim_pad=4,
    unify_by="height",
    allow_upscale=False,
    add_labels=True,
    label_style="plain",
    label_font_size=26,
    label_pad_x=10,
    label_pad_y=8,
):
    rows, cols = parse_layout(layout)

    if len(img_paths) == 0:
        raise ValueError("img_paths is empty.")

    if len(img_paths) > rows * cols:
        raise ValueError(
            f"Layout {layout} has only {rows * cols} slots, "
            f"but {len(img_paths)} images were provided."
        )

    # ===== load images =====
    imgs = [Image.open(p).convert("RGB") for p in img_paths]

    # ===== trim whitespace =====
    if trim_whitespace:
        imgs = [crop_white_border(img, pad=trim_pad) for img in imgs]

    # ===== resize without distortion =====
    target_value = compute_target_size(imgs, unify_by=unify_by)
    resized_imgs = [
        resize_preserve_aspect(img, target_value, unify_by, allow_upscale=allow_upscale)
        for img in imgs
    ]

    # ===== arrange into rows =====
    # If images < rows*cols, fill the rest with None
    slots = resized_imgs + [None] * (rows * cols - len(resized_imgs))

    row_heights = []
    row_widths = []

    for r in range(rows):
        row_imgs = slots[r * cols:(r + 1) * cols]
        valid_imgs = [img for img in row_imgs if img is not None]

        if len(valid_imgs) == 0:
            row_heights.append(0)
            row_widths.append(0)
            continue

        row_h = max(img.height for img in valid_imgs)
        row_w = sum(img.width for img in valid_imgs) + gap_x * (len(valid_imgs) - 1)

        row_heights.append(row_h)
        row_widths.append(row_w)

    canvas_w = margin_x * 2 + max(row_widths) if row_widths else margin_x * 2
    canvas_h = margin_y * 2 + sum(row_heights) + gap_y * max(0, rows - 1)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # ===== paste images =====
    positions = []
    y_cursor = margin_y

    for r in range(rows):
        row_imgs = slots[r * cols:(r + 1) * cols]
        valid_imgs = [img for img in row_imgs if img is not None]

        if len(valid_imgs) == 0:
            continue

        x_cursor = margin_x
        row_h = row_heights[r]

        for img in row_imgs:
            if img is None:
                continue

            # top align; if you want vertical centering, use:
            # y = y_cursor + (row_h - img.height) // 2
            x = x_cursor
            y = y_cursor

            canvas.paste(img, (x, y))
            positions.append((x, y, img.width, img.height))

            x_cursor += img.width + gap_x

        y_cursor += row_h + gap_y

    # ===== labels =====
    if add_labels:
        draw = ImageDraw.Draw(canvas)
        font = load_font(label_font_size)
        labels = [f"({chr(ord('a') + i)})" for i in range(len(resized_imgs))]

        for label, (x, y, w, h) in zip(labels, positions):
            tx = x + label_pad_x
            ty = y + label_pad_y

            if label_style == "box":
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                box_coords = [
                    tx - 6,
                    ty - 4,
                    tx + tw + 6,
                    ty + th + 5,
                ]
                draw.rectangle(box_coords, fill="white")

            draw.text((tx, ty), label, fill="black", font=font)

    return canvas


# ===== create and save =====
canvas = combine_images(
    img_paths=img_paths,
    layout=layout,
    margin_x=margin_x,
    margin_y=margin_y,
    gap_x=gap_x,
    gap_y=gap_y,
    trim_whitespace=trim_whitespace,
    trim_pad=trim_pad,
    unify_by=unify_by,
    allow_upscale=allow_upscale,
    add_labels=add_labels,
    label_style=label_style,
    label_font_size=label_font_size,
    label_pad_x=label_pad_x,
    label_pad_y=label_pad_y,
)

os.makedirs(os.path.dirname(out_png), exist_ok=True)

canvas.save(out_png)
canvas.save(out_pdf)

print(f"Saved to {out_png}")
print(f"Saved to {out_pdf}")