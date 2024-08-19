import torch
from phycv import PST_GPU, VEVID_GPU, PAGE_GPU
from phycv import utils
import cv2
import torchvision.transforms as transforms

def convert2torchTensor(image):
    transform = transforms.ToTensor() # Define a transform to convert the image to tensor
    image = transform(image) # Convert the image to PyTorch tensor
    image = torch.unsqueeze(image, dim=0) # Add the surrounding dimension back for comfyUI
    return image.permute(0, 2, 3, 1) # Permute doesn't change the input tensor : have to create another one

# Phase-Stretch Transform (PST)
class Pst_comfy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "phase_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "step": 0.01
                }),
                "warp_strength": ("INT", {
                    "default": 15,
                    "min": 1,
                    "step": 1
                }),
                "sigma_low_pass_filter": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "step": 0.01
                }),
                "thresh_min": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "step": 0.01
                }),
                "thresh_max": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "step": 0.01
                }),
                "morph_flag": ("BOOLEAN", {"default": True})
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("PST", "Kernel")
    FUNCTION = "pst"
    CATEGORY = "PhyCV"

    def pst(self, image, phase_strength, warp_strength, sigma_low_pass_filter, thresh_min, thresh_max, morph_flag):
        image = torch.squeeze(image)
        img = image.permute(2, 0, 1)

        # Extending PST_GPU so we can load from a tensor
        class PST_GPU_FROM_TENSOR(PST_GPU):
            def runFromTensor(self,img_array,S,W,sigma_LPF,thresh_min,thresh_max,morph_flag,):
                self.load_img(img_array=img_array)
                self.init_kernel(S, W)
                self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
                self.kernel_output = (torch.exp(-1j * self.pst_kernel))
                return self.pst_output, self.kernel_output

        """
            S (float): phase strength of PST
            W (float): warp of PST
            sigma_LPF (float): std of the low pass filter
            thresh_min (float): minimum thershold, we keep features < thresh_min
            thresh_max (float): maximum thershold, we keep features > thresh_max
            morph_flag (boolean): whether apply morphological operation
        """

        # run PST GPU version
        pst_gpu = PST_GPU_FROM_TENSOR(device=torch.device("cuda"))
        pst_output_gpu = pst_gpu.runFromTensor(
            img_array=img,
            S=phase_strength,
            W=warp_strength,
            sigma_LPF=sigma_low_pass_filter,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
            morph_flag=morph_flag,
        )
        transformed = torch.unsqueeze(pst_output_gpu[0], dim=0) # Needed for ComfyUI

        # Visualize the kernel
        kernel = pst_output_gpu[1]
        kernel = utils.normalize(torch.angle(kernel))
        kernel = kernel.cpu().numpy() # Convert image from Torch tensor to Numpy array
        kernel = cv2.normalize(kernel, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        kernel = cv2.cvtColor(kernel, cv2.COLOR_GRAY2RGB)

        return (transformed, convert2torchTensor(kernel))


# Vision Enhancement via Virtual diffraction and coherent Detection (VEViD)
class Vevid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "phase_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0,
                    "step": 0.01
                }),
                "spectral_phase_function_variance": ("FLOAT", {
                    "default": 0.01,
                    "min": 0,
                    "step": 0.01
                }),
                "regularization_term": ("FLOAT", {
                    "default": 0.16,
                    "min": 0.0,
                    "step": 0.01
                }),
                "phase_activation_gain": ("FLOAT", {
                    "default": 1.4,
                    "min": 0.0,
                    "step": 0.01
                }),
                "color_enhance": ("BOOLEAN", {
                    "default": False
                }),
                "lite": ("BOOLEAN", {
                    "default": False
                }),
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("Vevid", "Kernel",)
    FUNCTION = "Vevid"
    CATEGORY = "PhyCV"

    def Vevid(self, image, phase_strength, spectral_phase_function_variance, regularization_term, phase_activation_gain, color_enhance, lite):
        image = torch.squeeze(image)
        img = image.permute(2, 0, 1)

        """
        img_file (str): path to the image
        S (float): phase strength
        T (float): variance of the spectral phase function
        b (float): regularization term
        G (float): phase activation gain
        color (bool, optional): whether to run color enhancement. Defaults to False. 
        """

        class VEVID_GPU_FROM_TENSOR(VEVID_GPU):
            def runFromTensor(self, img_array, S, T, b, G, color, lite):
                self.load_img(img_array=img_array)
                self.init_kernel(S, T)
                self.apply_kernel(b, G, color, lite=lite)
                self.kernel_output = torch.exp(-1j * self.vevid_kernel)
                return self.vevid_output, self.kernel_output

        # run VEVID GPU version
        vevid_gpu = VEVID_GPU_FROM_TENSOR(device=torch.device("cuda"))
        vevid_output_gpu = vevid_gpu.runFromTensor(
            img_array=img,
            S=phase_strength, 
            T=spectral_phase_function_variance, 
            b=regularization_term, 
            G=phase_activation_gain, 
            color=color_enhance,
            lite=lite
        )

        image = torch.unsqueeze(vevid_output_gpu[0], dim=0) # Needed for comfy
        permuted = image.permute(0, 2, 3, 1)

        # Visualize the kernel
        kernel = vevid_output_gpu[1]
        kernel = utils.normalize(torch.angle(kernel))
        kernel = kernel.cpu().numpy() # Convert image from Torch tensor to Numpy array
        kernel = cv2.normalize(kernel, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        kernel = cv2.cvtColor(kernel, cv2.COLOR_GRAY2RGB)

        return (permuted, convert2torchTensor(kernel),)


# Phase-Stretch Adaptive Gradient-field Extractor (PAGE)
class Page:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mu_1": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "step": 0.01
                }),
                "mu_2": ("FLOAT", {
                    "default": 0.35,
                    "min": 0,
                    "step": 0.01
                }),
                "sigma_1": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "step": 0.01
                }),
                "sigma_2": ("FLOAT", {
                    "default": 0.7,
                    "min": 0,
                    "step": 0.01
                }),
                "mu_1_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "step": 0.01
                }),
                "mu_2_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "step": 0.01
                }),
                "sigma_LPF": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "step": 0.01
                }),
                "thresh_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0,
                    "step": 0.01
                }),
                "thresh_max": ("FLOAT", {
                    "default": 0.9,
                    "min": 0,
                    "step": 0.01
                }),
                "morph_flag": ("BOOLEAN", {"default": True})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "page"
    CATEGORY = "PhyCV"

    def page(self, image, mu_1, mu_2, sigma_1, sigma_2, mu_1_strength, mu_2_strength, sigma_LPF, thresh_min, thresh_max, morph_flag):
        image = torch.squeeze(image)
        img = image.permute(2, 0, 1)

        """
        mu_1 (float): Center frequency of a normal distributed passband filter ϕ1
        mu_2 (float):  Center frequency of log-normal  distributed passband filter ϕ2
        sigma_1 (float): Standard deviation of normal distributed passband filter ϕ1
        sigma_2 (float): Standard deviation of log-normal distributed passband filter ϕ2
        S1 (float): Phase strength of ϕ1
        S2 (float): Phase strength of ϕ2
        """

        class PAGE_FROM_TENSOR(PAGE_GPU):
            def runFromTensor(self, img_array, mu_1, mu_2, sigma_1, sigma_2, S1, S2, sigma_LPF, thresh_min,thresh_max, morph_flag,):
                self.load_img(img_array=img_array)
                self.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
                self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
                self.create_page_edge()
                return self.page_edge

        # run PAGE GPU version
        page_gpu = PAGE_FROM_TENSOR(direction_bins=10, device=torch.device("cuda"))
        page_edge_gpu = page_gpu.runFromTensor(
            img_array=img,
            mu_1=mu_1,
            mu_2=mu_2,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            S1=mu_1_strength,
            S2=mu_2_strength,
            sigma_LPF=sigma_LPF,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
            morph_flag=morph_flag,
        )

        page_edge_gpu = torch.unsqueeze(page_edge_gpu, dim=0)
        return (page_edge_gpu,)


NODE_CLASS_MAPPINGS = {
    "PST": Pst_comfy,
    "VEVID": Vevid,
    "PAGE": Page
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhyCV_PST": "PhyCV - Phase-Stretch Transform (PST)",
    "PhyCV_Vevid": "PhyCV - VEViD",
    "PhyCV_Page": "PhyCV - Page"
}