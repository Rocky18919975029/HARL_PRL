import os
import math
import argparse
from PIL import Image, ImageDraw, ImageFont


def excel_style_label(idx: int) -> str:
    """
    0 -> a, 1 -> b, ..., 25 -> z, 26 -> aa, ...
    """
    s = ""
    idx += 1
    while idx > 0:
        idx -= 1
        s = chr(ord("a") + (idx % 26)) + s
        idx //= 26
    return f"({s})"


def auto_grid(n):
    """
    Choose a near-square grid layout.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def infer_grid(n, layout, rows, cols):
    if layout == "horizontal":
        return 1, n
    elif layout == "vertical":
        return n, 1
    elif layout in ("grid", "auto"):
        if rows is not None and cols is not None:
            return rows, cols
        elif rows is not None:
            cols = math.ceil(n / rows)
            return rows, cols
        elif cols is not None:
            rows = math.ceil(n / cols)
            return rows, cols
        else:
            return auto_grid(n)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def resize_images(imgs, unify_by="none"):
    """
    unify_by:
        - none
        - height: resize all images to same height (min height)
        - width: resize all images to same width (min width)
    """
    if unify_by == "none":
        return imgs

    resized = []

    if unify_by == "height":
        target_h = min(img.height for img in imgs)
        for img in imgs:
            new_w = int(img.width * target_h / img.height)
            resized.append(img.resize((new_w, target_h), Image.LANCZOS))

    elif unify_by == "width":
        target_w = min(img.width for img in imgs)
        for img in imgs:
            new_h = int(img.height * target_w / img.width)
            resized.append(img.resize((target_w, new_h), Image.LANCZOS))
    else:
        raise ValueError(f"Unknown unify_by: {unify_by}")

    return resized


def get_font(font_size):
    candidates = [
        "DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except:
            continue
    return ImageFont.load_default()


def get_label_position(pos, x, y, w, h, text_w, text_h, pad=8):
    if pos == "tl":
        return x + pad, y + pad
    elif pos == "tr":
        return x + w - text_w - pad, y + pad
    elif pos == "bl":
        return x + pad, y + h - text_h - pad
    elif pos == "br":
        return x + w - text_w - pad, y + h - text_h - pad
    else:
        return x + pad, y + pad


def merge_images(
    img_paths,
    out_path="merged.png",
    layout="auto",
    rows=None,
    cols=None,
    gap=20,
    margin=20,
    bg_color="white",
    unify_by="none",
    add_labels=False,
    label_pos="tl",
    font_size=28,
):
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    imgs = resize_images(imgs, unify_by=unify_by)

    n = len(imgs)
    rows, cols = infer_grid(n, layout, rows, cols)

    cell_w = max(img.width for img in imgs)
    cell_h = max(img.height for img in imgs)

    canvas_w = margin * 2 + cols * cell_w + (cols - 1) * gap
    canvas_h = margin * 2 + rows * cell_h + (rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)
    font = get_font(font_size)

    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols

        cell_x = margin + c * (cell_w + gap)
        cell_y = margin + r * (cell_h + gap)

        paste_x = cell_x + (cell_w - img.width) // 2
        paste_y = cell_y + (cell_h - img.height) // 2

        canvas.paste(img, (paste_x, paste_y))

        if add_labels:
            label = excel_style_label(idx)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            lx, ly = get_label_position(
                label_pos, paste_x, paste_y, img.width, img.height, text_w, text_h
            )
            draw.text((lx, ly), label, fill="black", font=font)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    canvas.save(out_path, dpi=(300, 300))
    print(f"Saved merged image to: {out_path}")
    print(f"Layout: {rows} x {cols}")
    print(f"Canvas size: {canvas_w} x {canvas_h}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge arbitrary number of images with flexible layout."
    )

    parser.add_argument(
        "--imgs",
        type=str,
        nargs="+",
        required=True,
        help="Input image paths",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="merged.png",
        help="Output image path",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["horizontal", "vertical", "grid", "auto"],
        default="auto",
        help="Layout mode",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of rows (for grid/auto)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of cols (for grid/auto)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=20,
        help="Gap between images",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Outer margin",
    )
    parser.add_argument(
        "--bg_color",
        type=str,
        default="white",
        help="Background color",
    )
    parser.add_argument(
        "--unify_by",
        type=str,
        choices=["none", "height", "width"],
        default="none",
        help="Resize all images to same height or width before merging",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Add subplot labels like (a), (b), (c)",
    )
    parser.add_argument(
        "--label_pos",
        type=str,
        choices=["tl", "tr", "bl", "br"],
        default="tl",
        help="Label position",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=28,
        help="Label font size",
    )

    args = parser.parse_args()

    merge_images(
        img_paths=args.imgs,
        out_path=args.out,
        layout=args.layout,
        rows=args.rows,
        cols=args.cols,
        gap=args.gap,
        margin=args.margin,
        bg_color=args.bg_color,
        unify_by=args.unify_by,
        add_labels=args.labels,
        label_pos=args.label_pos,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()