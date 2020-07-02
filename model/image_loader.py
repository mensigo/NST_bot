from PIL import Image
import torch
import torchvision.transforms as transforms


def tensor2PIL(tensor, mean, std):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    image = torch.clamp(image, 0, 1)  
    image = transforms.ToPILImage()(image)
    return image


class Image_loader():
    def __init__(self, content_path_list, style_path_list,
                 img_size, device, mean, std, print_flg):
               
        self.device = device
        self.mean = mean
        self.std = std
        self.content_path_list = content_path_list
        self.style_path_list = style_path_list
        self.content_sizes = []
        self.content_imgs = []
        
        for p in self.content_path_list:
            
            image = Image.open(p)
            orig_w, orig_h = image.size

            if (max(orig_w,orig_h) > img_size):

                if (orig_w > orig_h):
                    new_w = img_size
                    ratio = img_size/float(orig_w)
                    new_h = int(round(ratio*orig_h)) 
                else:
                    new_h = img_size
                    ratio = img_size/float(orig_h)
                    new_w = int(round(ratio*orig_w))

                image = transforms.Resize((new_h,new_w), Image.LANCZOS)(image)
                if (print_flg):
                    print('Content of size {} is resized to size {}' \
                          .format((orig_w, orig_h), image.size))

            elif (print_flg):
                print('Content size is {}. No resize is needed.'.format((orig_w, orig_h)))

            image = transforms.ToTensor()(image).unsqueeze(0)
            self.content_sizes.append(image.numpy().shape[-2:])
            
            image = torch.FloatTensor(image)
            image = self.normalize(image)
            self.content_imgs.append(image)
                    
        # default
        self.curr_content_img = self.content_imgs[0]
        self.curr_content_size = self.content_sizes[0]
        
        # load style images
        self.load_style_images(self.style_path_list)
        
    def normalize(self, img):
        mean = self.mean.view(-1, 1, 1)
        std = self.std.view(-1, 1, 1)
        return (img - mean) / std
        
    def get_content(self):
        return self.curr_content_img.to(self.device, torch.float)
    
    def get_style(self, i):
        return self.style_imgs[i].to(self.device, torch.float)
        
    def set_content(self,i):
        self.curr_content_img = self.content_imgs[i]
        self.curr_content_size = self.content_sizes[i]
        self.load_style_images(self.style_path_list)
        
    def load_style_images(self, path_list):
        
        style_load = transforms.Compose([
            transforms.Resize(self.curr_content_size, Image.LANCZOS),
            transforms.ToTensor()])
        
        self.style_imgs = []
        for p in path_list:
            
            image = Image.open(p)
            image = style_load(image).unsqueeze(0)
            image = self.normalize(image)
            self.style_imgs.append(image)