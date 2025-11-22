import abc
import gin
import pandas as pd
import cv2
import numpy as np
import solutions.base_solution as BaseSolution
import solutions.FullyConnectedLayer as FCStack
import solutions.LongShortTermMemory as LSTMStack
import solutions.SelfAttention as SelfAttention
import solutions.Multi_Layer_Perception as MLPSolution
import os
import solutions.abc_solution as abc_solution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

@gin.configurable
class VisionTaskSolution(BaseSolution):
    """A general solution for vision based tasks."""

    def __init__(self,
                 image_size,
                 query_dim,
                 output_dim,
                 output_activation,
                 num_hiddens,
                 l2_coefficient,
                 patch_size,
                 patch_stride,
                 top_k,
                 data_dim,
                 activation,
                 normalize_positions=False,
                 use_lstm_controller=False,
                 show_overplot=False):
        super(VisionTaskSolution, self).__init__()
        self._image_size = image_size
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._top_k = top_k
        self._l2_coefficient = l2_coefficient
        self._show_overplot = show_overplot
        self._normalize_positions = normalize_positions
        self._screen_dir = None
        self._img_ix = 1
        self._raw_importances = []

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        n = int((image_size - patch_size) / patch_stride + 1)
        offset = self._patch_size // 2
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * patch_stride
            for j in range(n):
                patch_center_col = offset + j * patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        self._patch_centers = torch.tensor(patch_centers).float()

        num_patches = n ** 2
        print('num_patches = {}'.format(num_patches))
        self._attention = SelfAttention(
            data_dim=data_dim * self._patch_size ** 2,
            dim_q=query_dim,
        )
        self._layers.extend(self._attention.layers)

        self._mlp_solution = MLPSolution(
            input_dim=self._top_k * 2,
            num_hiddens=num_hiddens,
            activation=activation,
            output_dim=output_dim,
            output_activation=output_activation,
            l2_coefficient=l2_coefficient,
            use_lstm=use_lstm_controller,
        )
        self._layers.extend(self._mlp_solution.layers)

        print('Number of parameters: {}'.format(
            self.get_num_params_per_layer()))

    def _get_output(self, inputs, update_filter):

        # ob.shape = (h, w, c)
        ob = self._transform(inputs).permute(1, 2, 0)
        # print(ob.shape)
        h, w, c = ob.size()
        patches = ob.unfold(
            0, self._patch_size, self._patch_stride).permute(0, 3, 1, 2)
        patches = patches.unfold(
            2, self._patch_size, self._patch_stride).permute(0, 2, 1, 4, 3)
        patches = patches.reshape((-1, self._patch_size, self._patch_size, c))

        # flattened_patches.shape = (1, n, p * p * c)
        flattened_patches = patches.reshape(
            (1, -1, c * self._patch_size ** 2))
        # attention_matrix.shape = (1, n, n)
        attention_matrix = self._attention(flattened_patches)
        # patch_importance_matrix.shape = (n, n)
        patch_importance_matrix = torch.softmax(
            attention_matrix.squeeze(), dim=-1)
        # patch_importance.shape = (n,)
        patch_importance = patch_importance_matrix.sum(dim=0)
        # extract top k important patches
        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self._top_k]

        centers = self._patch_centers[top_k_ix]

        # Overplot.
        if self._show_overplot:
            task_image = ob.numpy().copy()
            patch_importance_copy = patch_importance.numpy().copy()


            if self._screen_dir is not None:
                # Save the original screen.
                img_filepath = os.path.join(
                    self._screen_dir, 'orig_{0:04d}.png'.format(self._img_ix))
                cv2.imwrite(img_filepath, inputs[:, :, ::-1])
                # Save the scaled screen.
                img_filepath = os.path.join(
                    self._screen_dir, 'scaled_{0:04d}.png'.format(self._img_ix))
                cv2.imwrite(
                    img_filepath,
                    (task_image * 255).astype(np.uint8)[:, :, ::-1]
                )
                # Save importance vectors.
                dd = {
                    'step': self._img_ix,
                    'importance': patch_importance_copy.tolist(),
                }
                self._raw_importances.append(dd)

                if self._img_ix % 20 == 0:
                    csv_path = os.path.join(self._screen_dir, 'importances.csv')
                    pd.DataFrame(self._raw_importances).to_csv(
                        csv_path, index=False
                    )

            white_patch = np.ones(
                (self._patch_size, self._patch_size, 3))
            half_patch_size = self._patch_size // 2
            for i, center in enumerate(centers):
                row_ss = int(center[0]) - half_patch_size
                row_ee = int(center[0]) + half_patch_size + 1
                col_ss = int(center[1]) - half_patch_size
                col_ee = int(center[1]) + half_patch_size + 1
                ratio = 1.0 * i / self._top_k
                task_image[row_ss:row_ee, col_ss:col_ee] = (
                        ratio * task_image[row_ss:row_ee, col_ss:col_ee] +
                        (1 - ratio) * white_patch)
            task_image = cv2.resize(
                task_image, (task_image.shape[0] * 5, task_image.shape[1] * 5))
            cv2.imshow('Overplotting', task_image[:, :, [2, 1, 0]])
            cv2.waitKey(1)

            if self._screen_dir is not None:
                # Save the scaled screen.
                img_filepath = os.path.join(
                    self._screen_dir, 'att_{0:04d}.png'.format(self._img_ix))
                cv2.imwrite(
                    img_filepath,
                    (task_image * 255).astype(np.uint8)[:, :, ::-1]
                )

            self._img_ix += 1

        centers = centers.flatten(0, -1)
        if self._normalize_positions:
            centers = centers / self._image_size

        return self._mlp_solution.get_output(centers)

    def reset(self):
        self._selected_patch_centers = []
        self._value_network_input_images = []
        self._accumulated_gradients = None
        self._mlp_solution.reset()
        self._img_ix = 1
        self._raw_importances = []

    def set_log_dir(self, folder):
        self._screen_dir = folder
        if not os.path.exists(self._screen_dir):
            os.makedirs(self._screen_dir)