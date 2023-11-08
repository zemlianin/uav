from PIL import Image
import argparse
import logging
import os


def add_margin(pil_img, top: int, right: int, bottom: int, left: int, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def dynamic_crop(pil_img):
    crop_left = 0
    while True:
        width, height = pil_img.size
        edge_values = []
        for h in range(height):
            edge_values.append(sum(pil_img.getpixel((0, h))) / 3)
        if max(edge_values) <= 5 or min(edge_values) >= 250 or \
                sum(edge_values) / len(edge_values) < 1 or sum(edge_values) / len(edge_values) > 254:
            pil_img = pil_img.crop((1, 0, width, height))
        else:
            break
        crop_left += 1

    crop_right = 0
    while True:
        width, height = pil_img.size
        edge_values = []
        for h in range(height):
            edge_values.append(sum(pil_img.getpixel((width - 1, h))) / 3)
        if max(edge_values) <= 5 or min(edge_values) >= 250 or \
                sum(edge_values) / len(edge_values) < 1 or sum(edge_values) / len(edge_values) > 254:
            pil_img = pil_img.crop((0, 0, width - 1, height))
        else:
            break
        crop_right += 1

    crop_top = 0
    while True:
        width, height = pil_img.size
        edge_values = []
        for w in range(width):
            edge_values.append(sum(pil_img.getpixel((w, 0))) / 3)
        if max(edge_values) <= 5 or min(edge_values) >= 250 or \
                sum(edge_values) / len(edge_values) < 1 or sum(edge_values) / len(edge_values) > 254:
            pil_img = pil_img.crop((0, 1, width, height))
        else:
            break
        crop_top += 1

    crop_bottom = 0
    while True:
        width, height = pil_img.size
        edge_values = []
        for w in range(width):
            edge_values.append(sum(pil_img.getpixel((w, height - 1))) / 3)
        if max(edge_values) <= 5 or min(edge_values) >= 250 or \
                sum(edge_values) / len(edge_values) < 1 or sum(edge_values) / len(edge_values) > 254:
            pil_img = pil_img.crop((0, 0, width, height - 1))
        else:
            break
        crop_bottom += 1

    return pil_img, crop_left, crop_right, crop_top, crop_bottom


def dynamic_padding(pil_img):
    width, height = pil_img.size

    if 640 > height and 640 > width:
        pad_left = (640 - width) // 2
        pad_right = 640 - width - pad_left
        pad_top = (640 - height) // 2
        pad_bottom = 640 - height - pad_top
    elif width > height:
        pad_left, pad_right = 0, 0
        pad_top = (width - height) // 2
        pad_bottom = width - height - pad_top
    elif width < height:
        pad_top, pad_bottom = 0, 0
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
    else:
        return pil_img

    padded_image = add_margin(
        pil_img, pad_top, pad_right, pad_bottom, pad_left, (0, 0, 0))

    return padded_image, pad_left, pad_right, pad_top, pad_bottom


def process_directories(old_directory: str, new_directory: str) -> None:
    got = set()
    for path in os.listdir(os.path.join(old_directory, 'images')):
        try:
            im = Image.open(os.path.join(old_directory, 'images'))
            im2, left_minus, right_minus, top_minus, bottom_minus = dynamic_crop(im)
            # if im2.width <= 320 or im2.height <= 320:
                # print(f'image {path} bad quality')
            file = path.split('.')[0]
            if file in got and not file.startswith('youtube'):
                print(f'image {path} got')
            else:
                got.add(file)
            im2, left_plus, _, top_plus, _ = dynamic_padding(im2)
            im2.save(os.path.join(new_directory, 'images'))
            with open(os.path.splitext(os.path.join(os.path.join(old_directory, 'labels'), path))[0] + '.txt', 'r') as f:
                with open(os.path.splitext(os.path.join(os.path.join(new_directory, 'labels'), path))[0] + '.txt', 'w') as f_w:
                    lines = f.readlines()
                    new_lines = []
                    for line in lines:
                        new_line = line.split(' ')[0]
                        new_line += ' '
                        new_line += str((float(line.split(' ')[1]) * im.width + left_plus - left_minus) / im2.width)
                        new_line += ' '
                        new_line += str((float(line.split(' ')[2]) * im.height + top_plus - top_minus) / im2.height)
                        new_line += ' '
                        new_line += str((float(line.split(' ')[3]) * im.width) / im2.width)
                        new_line += ' '
                        new_line += str((float(line.split(' ')[4]) * im.height) / im2.height)
                        new_line += '\n'
                        new_lines.append(new_line)
                    f_w.writelines(new_lines)
        except Exception as e:
            logging.WARNING(f'image {path} bad: {e}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("old_directory",
                        help="directory with not preprocessed images",
                        type=str)

    parser.add_argument("new_directory",
                        help="directory after preprocessing images",
                        type=str)

    parser.add_argument("-q",
                        "--quiet-mode",
                        default=False,
                        help="do not logging.info() messages",
                        action="store_true")

def setup_logging(quiet_mode):
    logging.basicConfig(level=logging.WARNING if quiet_mode else logging.INFO,
                        format="%(message)s")


def main() -> None:
    args = parse_args()
    setup_logging(args.quiet_mode)
    process_directories(args.old_directory, args.new_directory)


main()
