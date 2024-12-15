import base64
import io
import math

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


def pil_image_to_base64_jpeg(rgb_image: Image):
    # In-memory buffer for the JPEG image
    buffered = io.BytesIO()

    # Save as JPEG
    rgb_image.save(buffered, format="JPEG")

    # Encode as base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


def image_file_to_base64_jpeg(image_path: str) -> str:
    """
    Reads an image from disk, converts it to JPEG, and encodes it as a base64 string.
    """
    # Open the image
    rgb_image = Image.open(image_path).convert("RGB")

    # In-memory buffer for the JPEG image
    buffered = io.BytesIO()

    # Save as JPEG
    rgb_image.save(buffered, format="JPEG")

    # Encode as base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


def base64_to_pil_image(base64_string: str):
    # Decode the base64 string into binary data
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO stream from the binary data
    image_stream = io.BytesIO(image_data)

    # Open the image stream with PIL
    pil_image = Image.open(image_stream)

    return pil_image


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """
    Source: https://github.com/datitran/object_detector_app/blob/master/object_detection/utils/visualization_utils.py#L122
    Adds a bounding box to an image.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = top
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        _, _, text_width, text_height = font.getbbox(display_str)

        margin = math.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin
