from PIL import Image
import argparse
import logging
import os
import glob

req_width=512
req_height=512

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
        if max(edge_values) <= 10 or min(edge_values) >= 245 or \
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
        if max(edge_values) <= 10 or min(edge_values) >= 245 or \
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
        if max(edge_values) <= 10 or min(edge_values) >= 245 or \
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
        if max(edge_values) <= 10 or min(edge_values) >= 245 or \
                sum(edge_values) / len(edge_values) < 1 or sum(edge_values) / len(edge_values) > 254:
            pil_img = pil_img.crop((0, 0, width, height - 1))
        else:
            break
        crop_bottom += 1

    pil_img = pil_img.resize((min(pil_img.width, 512), min(pil_img.height, 512)))

    print(f"{pil_img}, {crop_left}, {crop_right}, {crop_top}, {crop_bottom}")

    return pil_img, crop_left, crop_right, crop_top, crop_bottom


def dynamic_padding(pil_img):
    width, height = pil_img.size

    if req_height > height and req_width > width:
        pad_left = (req_width - width) // 2
        pad_right = req_width - width - pad_left
        pad_top = (req_height - height) // 2
        pad_bottom = req_height - height - pad_top
    elif width > height:
        pad_left, pad_right = 0, 0
        pad_top = (width - height) // 2
        pad_bottom = width - height - pad_top
    elif width < height:
        pad_top, pad_bottom = 0, 0
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
    else:
        return pil_img, 0, 0, 0, 0

    padded_image = add_margin(
        pil_img, pad_top, pad_right, pad_bottom, pad_left, (0, 0, 0))

    return padded_image, pad_left, pad_right, pad_top, pad_bottom


def process_directories(old_directory: str, new_directory: str) -> None:
    got = set()
    
    for path in glob.glob(os.path.join(old_directory, 'images') + '/*'):
        try:
            im = Image.open(path)
            print(path)
            response = dynamic_crop(im)
            print(response)
            im2=response[0]
            left_minus=response[1]
            right_minus=response[2]
            top_minus=response[3]
            bottom_minus=response[4]
            # if im2.width <= 320 or im2.height <= 320:
                # print(f'image {path} bad quality')
            file = path.split('.')[0]
            if file in got and not file.startswith('youtube'):
                print(f'image {path} got')
            else:
                got.add(file)
            response=dynamic_padding(im2)
            print(response)

            im2, left_plus, t, top_plus, y = dynamic_padding(im2)

            new_path=path.replace(old_directory, new_directory)
            im2.save(new_path)
            print(path)
            name = path.replace("old_photoes\images\\","")
            print(name)
            with open(os.path.splitext(os.path.join(os.path.join(old_directory, 'labels'), name))[0] + '.txt', 'r') as f:
                with open(os.path.splitext(os.path.join(os.path.join(new_directory, 'labels'), name))[0] + '.txt', 'w') as f_w:
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

def setup_logging():
    logging.basicConfig(level=logging.WARNING,
                        format="%(message)s")


def main() -> None:
    #args = parse_args()
   # print(args)
    setup_logging()
    process_directories("old_photoes", "new_photoes")


main()
