import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to inference control net')
    parser.add_argument('--image_path', default='', help='Path to image', type=str)
    parser.add_argument('--prompt', default='', help='Prompt', type=str)
    parser.add_argument('--img_size', default=512, help='Image size', type=int)
    parser.add_argument('--steps', default=20, help='Ddim steps', type=int)
    parser.add_argument('--num_sample', default=1, help='Number of samples', type=int)
    parser.add_argument('--output_folder', default="", help='Output folder', type=str)
    parser.add_argument('--prefix_output', default="", help='Prefix output images', type=str)

    return parser.parse_args()