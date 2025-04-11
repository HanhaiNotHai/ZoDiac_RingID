import argparse
import itertools
import os
import time

import torch
import yaml
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image
from torch import Tensor
from tqdm import tqdm, trange

from RingID.inverse_stable_diffusion import InversableStableDiffusionPipeline
from RingID.utils import *
from ZoDiac.loss.loss import LossProvider
from ZoDiac.loss.pytorch_ssim import ssim
from ZoDiac.main.utils import compute_psnr, get_img_tensor, save_img, watermark_prob
from ZoDiac.main.wmdiffusion import WMDetectStableDiffusionPipeline
from ZoDiac.main.wmpatch import GTWatermark, GTWatermarkMulti


def parse_args():
    parser = argparse.ArgumentParser(description='multiple-key identification')
    parser.add_argument('--run_name', default='ZoDiaz_RingID')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--reference_model', default='ViT-g-14')
    parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')

    group = parser.add_argument_group('hyperparameters')
    parser.add_argument('--general_seed', type=int, default=42)
    parser.add_argument('--watermark_seed', type=int, default=5)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--ring_width', default=1, type=int)
    parser.add_argument('--num_inmost_keys', default=2, type=int)
    parser.add_argument('--ring_value_range', default=64, type=int)

    parser.add_argument('--save_generated_imgs', type=int, default=1)
    parser.add_argument('--save_root_dir', type=str, default='./runs')

    group = parser.add_argument_group('trials parameters')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--trials', type=int, default=100, help='total number of trials to run')
    parser.add_argument(
        '--fix_gt',
        type=int,
        default=1,
        help='use watermark after discarding the imag part on space domain as gt.',
    )
    parser.add_argument('--time_shift', type=int, default=1, help='use time-shift')
    parser.add_argument(
        '--time_shift_factor',
        type=float,
        default=1.0,
        help='factor to scale the value after time-shift',
    )
    parser.add_argument(
        '--assigned_keys',
        type=int,
        default=-1,
        help='number of assigned keys, -1 for all possible kyes',
    )
    parser.add_argument(
        '--channel_min',
        type=int,
        default=1,
        help='only for heterogeous watermark, when match gt, take min among channels as the result',
    )

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    return args


def get_init_latent(
    img_tensor, pipe: WMDetectStableDiffusionPipeline, text_embeddings, guidance_scale=1.0
):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    return reversed_latents


def binary_search_theta(
    gt_img_tensor, wm_img_tensor, threshold, lower=0.0, upper=1.0, precision=1e-6, max_iter=1000
):
    for i in range(max_iter):
        mid_theta = (lower + upper) / 2
        img_tensor = (gt_img_tensor - wm_img_tensor) * mid_theta + wm_img_tensor
        ssim_value = ssim(img_tensor, gt_img_tensor).item()

        if ssim_value <= threshold:
            lower = mid_theta
        else:
            upper = mid_theta
        if upper - lower < precision:
            break
    return lower


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    ######################
    # ZoDiac
    ######################
    with open('ZoDiac/example/config/config.yaml', 'r') as file:
        cfgs = yaml.safe_load(file)

    # hyperparameter
    ssim_threshold = cfgs['ssim_threshold']

    if cfgs['w_type'] == 'single':
        wm_pipe = GTWatermark(
            device,
            w_channel=cfgs['w_channel'],
            w_radius=cfgs['w_radius'],
            generator=torch.Generator(device).manual_seed(cfgs['w_seed']),
        )
    elif cfgs['w_type'] == 'multi':
        wm_pipe = GTWatermarkMulti(
            device,
            w_settings=cfgs['w_settings'],
            generator=torch.Generator(device).manual_seed(cfgs['w_seed']),
        )

    scheduler = DDIMScheduler.from_pretrained(cfgs['model_id'], subfolder="scheduler")
    pipe: WMDetectStableDiffusionPipeline = WMDetectStableDiffusionPipeline.from_pretrained(
        cfgs['model_id'], scheduler=scheduler
    )
    pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    empty_text_embeddings = pipe.get_text_embedding('')

    totalLoss = LossProvider(cfgs['loss_weights'], device)

    ######################
    # RingID
    ######################
    # args = parse_args()

    # scheduler = DPMSolverMultistepScheduler.from_pretrained(
    #     cfgs['model_id'], subfolder='scheduler'
    # )
    # pipe_r = InversableStableDiffusionPipeline(
    #     pipe.vae,
    #     pipe.text_encoder,
    #     pipe.tokenizer,
    #     pipe.unet,
    #     scheduler,
    #     pipe.safety_checker,
    #     pipe.feature_extractor,
    #     pipe.image_encoder,
    #     pipe.config.requires_safety_checker,
    # )
    # pipe_r.set_progress_bar_config(leave=False)

    # if args.channel_min:
    #     assert len(HETER_WATERMARK_CHANNEL) > 0

    # eval_methods = [
    #     {
    #         'Distance': 'L1',
    #         'Metrics': '|a-b|        ',
    #         'func': get_distance,
    #         'kwargs': {'p': 1, 'mode': 'complex', 'channel_min': args.channel_min},
    #     }
    # ]

    # base_latents = pipe_r.get_random_latents()
    # base_latents = base_latents.to(torch.float64)
    # original_latents_shape = base_latents.shape
    # sing_channel_ring_watermark_mask = torch.tensor(
    #     ring_mask(size=original_latents_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF)
    # )

    # # get heterogeneous watermark mask
    # if len(HETER_WATERMARK_CHANNEL) > 0:
    #     single_channel_heter_watermark_mask = torch.tensor(
    #         ring_mask(
    #             size=original_latents_shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF
    #         )  # TODO: change to whole mask
    #     )
    #     heter_watermark_region_mask = (
    #         single_channel_heter_watermark_mask.unsqueeze(0)
    #         .repeat(len(HETER_WATERMARK_CHANNEL), 1, 1)
    #         .to(device)
    #     )

    # watermark_region_mask = []
    # for channel_idx in WATERMARK_CHANNEL:
    #     if channel_idx in RING_WATERMARK_CHANNEL:
    #         watermark_region_mask.append(sing_channel_ring_watermark_mask)
    #     else:
    #         watermark_region_mask.append(single_channel_heter_watermark_mask)
    # watermark_region_mask = torch.stack(watermark_region_mask).to(device)  # [C, 64, 64]

    # # ###### Make RingID pattern
    # single_channel_num_slots = RADIUS - RADIUS_CUTOFF
    # key_value_list = [
    #     [
    #         list(combo)
    #         for combo in itertools.product(
    #             np.linspace(
    #                 -args.ring_value_range, args.ring_value_range, args.num_inmost_keys
    #             ).tolist(),
    #             repeat=len(RING_WATERMARK_CHANNEL),
    #         )
    #     ]
    #     for _ in range(single_channel_num_slots)
    # ]
    # key_value_combinations = list(itertools.product(*key_value_list))

    # # random select from all possible value combinations, then generate patterns for selected ones.
    # if args.assigned_keys > 0:
    #     assert args.assigned_keys <= len(key_value_combinations)
    #     key_value_combinations = random.sample(key_value_combinations, k=args.assigned_keys)
    # Fourier_watermark_pattern_list = [
    #     make_Fourier_ringid_pattern(
    #         device,
    #         list(combo),
    #         base_latents,
    #         radius=RADIUS,
    #         radius_cutoff=RADIUS_CUTOFF,
    #         ring_watermark_channel=RING_WATERMARK_CHANNEL,
    #         heter_watermark_channel=HETER_WATERMARK_CHANNEL,
    #         heter_watermark_region_mask=(
    #             heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL) > 0 else None
    #         ),
    #     )
    #     for _, combo in enumerate(key_value_combinations)
    # ]

    # ring_capacity = len(Fourier_watermark_pattern_list)

    # if args.fix_gt:
    #     Fourier_watermark_pattern_list = [
    #         fft(ifft(Fourier_watermark_pattern).real)
    #         for Fourier_watermark_pattern in Fourier_watermark_pattern_list
    #     ]

    # if args.time_shift:
    #     for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
    #         # Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * args.time_shift_factor)
    #         Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(
    #             torch.fft.fftshift(
    #                 ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2)
    #             )
    #         )

    # key_indices_to_evaluate = np.random.choice(
    #     ring_capacity, size=args.trials, replace=True
    # ).tolist()

    dataset = 'dataset/DIV2K/DIV2K_train_HR'
    wm_path = dataset + '_wm'
    os.makedirs(wm_path, exist_ok=True)
    test_num = 100
    t = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, time.strftime(f'%Y-%m-%d-%H:%M:%S'))

    ori_ssim_values = []
    ssim_values = []
    psnr_values = []
    det_probs = []
    img_paths = []
    for imagename in tqdm(os.listdir(dataset)[:test_num]):
        image_path = os.path.join(dataset, imagename)
        gt_img_tensor: Tensor = t(Image.open(image_path))
        gt_img_tensor = gt_img_tensor.unsqueeze(0).to(device)

        # Step 1: Get init noise
        init_latents_approx = get_init_latent(gt_img_tensor, pipe, empty_text_embeddings)

        # Step 2: prepare training
        init_latents = init_latents_approx.detach().clone()
        init_latents.requires_grad = True
        optimizer = torch.optim.Adam([init_latents], lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3)

        # Step 3: train the init latents
        with trange(cfgs['iters'], leave=False) as pbar:
            for i in pbar:
                init_latents_wm = wm_pipe.inject_watermark(init_latents)
                pred_img_tensor = pipe(
                    '',
                    guidance_scale=1.0,
                    num_inference_steps=50,
                    output_type='tensor',
                    use_trainable_latents=True,
                    init_latents=init_latents_wm,
                ).images
                loss: Tensor = totalLoss(
                    pred_img_tensor, gt_img_tensor, init_latents_wm, wm_pipe, pbar
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # save watermarked image
                if (i + 1) in cfgs['save_iters']:
                    path = os.path.join(wm_path, f"{imagename.split('.')[0]}_{i+1}.png")
                    save_img(path, pred_img_tensor, pipe)

        wm_img_path = os.path.join(
            wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}.png"
        )
        wm_img_tensor = get_img_tensor(wm_img_path, device)
        ori_ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()
        ori_ssim_values.append(ori_ssim_value)

        optimal_theta = binary_search_theta(
            gt_img_tensor, wm_img_tensor, ssim_threshold, precision=0.01
        )
        img_tensor = (gt_img_tensor - wm_img_tensor) * optimal_theta + wm_img_tensor

        ssim_value = ssim(img_tensor, gt_img_tensor).item()
        psnr_value = compute_psnr(img_tensor, gt_img_tensor)
        det_prob = 1 - watermark_prob(img_tensor, pipe, wm_pipe, empty_text_embeddings)
        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)
        det_probs.append(det_prob)

        path = os.path.join(
            wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png"
        )
        img_paths.append(path)
        save_img(path, img_tensor, pipe)

    with open(result_path, 'a') as f:
        f.write(f'original ssim {sum(ori_ssim_values) / test_num}\n')
        f.write(f'ssim {sum(ssim_values) / test_num}\n')
        f.write(f'psnr {sum(psnr_values) / test_num}\n')
        f.write(f'detect probility {sum(det_probs) / test_num}\n')

    img2img = StableDiffusionImg2ImgPipeline(
        pipe.vae,
        pipe.text_encoder,
        pipe.tokenizer,
        pipe.unet,
        pipe.scheduler,
        pipe.safety_checker,
        pipe.feature_extractor,
        pipe.image_encoder,
        pipe.config.requires_safety_checker,
    )
    img2img.set_progress_bar_config(leave=False)

    for strength_i in trange(1, 10):
        strength = strength_i / 10
        det_probs = []
        for img_path in tqdm(img_paths, leave=False):
            image = Image.open(img_path)
            distorted_img_tensor = img2img('', image, strength, output_type='pt').images
            det_prob = 1 - watermark_prob(
                distorted_img_tensor, pipe, wm_pipe, empty_text_embeddings
            )
            det_probs.append(det_prob)
        with open(result_path, 'a') as f:
            f.write(f'strength {strength} detect probility {sum(det_probs) / test_num}\n')


if __name__ == '__main__':
    main()
