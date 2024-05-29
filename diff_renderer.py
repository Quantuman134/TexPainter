import torch
from pytorch3d import renderer
from texture import Texture
from diff_shader import DiffShader

class DiffRenderer:
    def __init__(self, offset=[0.0, 0.0, 0.0], device='cpu') -> None:
        self.device = device
        self.offset = torch.tensor([offset]) #to center of object
        self.rasterization_setting()
        self.camera_setting(offset=self.offset)
        self.light_setting()

    def rendering(self, mesh_data, diff_tex:Texture, shading_method='phong', background_map=None):
        mesh_renderer = renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_setting
            ),
            shader=DiffShader(
                diff_tex=diff_tex,
                device=self.device,
                cameras=self.cameras,
                lights=self.lights,
                mesh_data=mesh_data,
                shading_method=shading_method,
                background_map=background_map
            )
        )

        return mesh_renderer(mesh_data['mesh_obj'])        

    def camera_setting(self, dist=2.0, elev=0, azim=135, fov=60, offset=torch.tensor([[0, 0, 0]])):
        R, T = renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim)
        T += offset
        self.cameras = renderer.FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)

    def rasterization_setting(self, image_size=512, blur_radius=0.0, face_per_pixel=1):
        self.raster_setting = renderer.RasterizationSettings(image_size=image_size, blur_radius=blur_radius, faces_per_pixel=face_per_pixel, cull_backfaces=True)

    def light_setting(self, directions=[[1.0, 1.0, 1.0]]):
        self.lights = renderer.DirectionalLights(direction=directions, device=self.device)


def test():
    import torch
    from pytorch3d import io
    import matplotlib.pyplot as plt 
    import numpy as np
    from texture import Texture
    from PIL import Image
    from torchvision import transforms
    import utils

    renderer = DiffRenderer(device=utils.device)
    #mesh_path = "./Assets/3D_Model/Basketball/basketball.obj"
    mesh_path = './Assets/Cow.obj'
    image_path = "./Assets/cow_texture.png"
    save_path = "./temp"

    # differentiable texture
    diff_tex = Texture(size=(512, 512), is_latent=False, device=utils.device)
    image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    diff_tex.set_image(image_tensor)

    # mlp texture
    #diff_tex = NeuralTextureField(width=32, depth=2, input_dim=3, brdf=True)
    #diff_tex.tex_load(mlp_path)

    mesh_obj = io.load_objs_as_meshes([mesh_path], device=utils.device)

    verts_packed = mesh_obj.verts_packed()
    
    verts_max = verts_packed.max(dim=0).values
    verts_min = verts_packed.min(dim=0).values
    max_length = (verts_max - verts_min).max().item()
    center = (verts_max + verts_min)/2

    verts_list = mesh_obj.verts_list()
    verts_list[:] = [(verts_obj - center)/max_length for verts_obj in verts_list] #[-0.5, 0.5]
    mesh_obj._verts_packed = (verts_packed - center)/max_length

    verts, faces, aux = io.load_obj(mesh_path, device=utils.device)
    verts = (verts - center)/max_length

    mesh_data = {'mesh_obj': mesh_obj,'verts': verts, 'faces': faces, 'aux': aux}

    offset = torch.tensor([-0.0, 0.0, -0.0])
    renderer.camera_setting(dist=1.5, elev=15, azim=180, offset=offset, fov=45)
    renderer.rasterization_setting(image_size=512)
    #renderer.light_setting(directions=[[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]], 
    #                       intensities=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], multi_lights=True)

    background_map = torch.ones((1, 3, 512, 512), dtype=torch.float32, device=utils.device) * 0.5
    background_map[:, 2, :, :] = 0.0
    image_tensor = renderer.rendering(mesh_data=mesh_data, diff_tex=diff_tex, shading_method='phong', background_map=background_map).images
    image_array = image_tensor[0, :, :, 0:3].cpu().detach().numpy()
    image_array = np.clip(image_array, 0, 1)
    plt.imshow(image_array)
    plt.show()
    utils.save_img_tensor(image_tensor[:, :, :, 0:3].permute(0, 3, 1, 2), save_dir=save_path + '/back.png')


if __name__ == "__main__":
    test()

