import os
import click
import sys

sys.path.append("./stylegan3")
import dnnlib
import legacy
import numpy as np
import torch
import PIL.Image
from typing import List, Optional, Tuple


def parse_range(s: str) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.'''
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_vec2(s: str) -> Tuple[float, float]:
    '''Parse a float2 vector'''
    elems = s.split(',')
    return (float(elems[0]), float(elems[1]))


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0',
              show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num', type=int, help='Number of images to generate', default=1, show_default=True)
@click.option('--batch-size', type=int, help='Batch size for generation', default=1, show_default=True)
@click.option('--direction-path', type=str, help='Path to direction vector file', default=None)
@click.option('--coeff', type=float, help='Coefficient for direction vector', default=None)
@click.option('--repeat-generation', type=bool, help='Repeat generation using saved latents')
def generate_manipulated_images(
        network_pkl: str,
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        translate: Tuple[float, float],
        rotate: float,
        class_idx: Optional[int],
        num: int,
        batch_size: int,
        direction_path: Optional[str],
        coeff: Optional[float],
        repeat_generation: bool
):
    """Generate images using pretrained network pickle with optional manipulations."""

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    if direction_path is not None:
        direction = torch.from_numpy(np.load(direction_path)).to(device).unsqueeze(0)

    # Labels
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('Warning: --class=lbl ignored when running on an unconditional network')

    # Generate images
    for seed_idx in range(0, num, batch_size):
        print(f'Generating images {seed_idx + 1}-{min(seed_idx + batch_size, num)} / {num} ...')

        z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi)

        if repeat_generation:
            for j in range(batch_size):
                w[j] = torch.from_numpy(np.load(f'{outdir}/latent_{seed_idx + j}.npy')).to(device)

        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        if direction_path is not None and coeff is not None:
            pos_w = w.clone()
            neg_w = w.clone()
            pos_w[:,:8] += coeff * direction[:,:8]
            neg_w[:,:8] -= coeff * direction[:,:8]

            pos_img = G.synthesis(pos_w, noise_mode=noise_mode)
            neg_img = G.synthesis(neg_w, noise_mode=noise_mode)

            pos_img = (pos_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            neg_img = (neg_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            for j in range(batch_size):
                PIL.Image.fromarray(pos_img[j].cpu().numpy(), 'RGB').save(f'{outdir}/{seed_idx + j:04d}_pos{coeff}.png')
                PIL.Image.fromarray(neg_img[j].cpu().numpy(), 'RGB').save(f'{outdir}/{seed_idx + j:04d}_neg{coeff}.png')

        img = G.synthesis(w, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        for j in range(batch_size):
            PIL.Image.fromarray(img[j].cpu().numpy(), 'RGB').save(f'{outdir}/{seed_idx + j:04d}.png')
            if not repeat_generation:
                np.save(f'{outdir}/latent_{seed_idx + j}.npy', w[j].cpu().numpy())


if __name__ == "__main__":
    generate_manipulated_images()