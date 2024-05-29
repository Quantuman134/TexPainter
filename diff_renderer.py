import torch
from pytorch3d import renderer

class DiffRenderer:
    def __init__(self, offset=[0.0, 0.0, 0.0], device='cpu') -> None:
        self.device = device
        self.offset = torch.tensor([offset]) #to center of object
        self.rasterization_setting()
        self.camera_setting(offset=self.offset)
        self.light_setting()

    def camera_setting(self, dist=2.0, elev=0, azim=135, fov=60, offset=torch.tensor([[0, 0, 0]])):
        R, T = renderer.look_at_view_transform(dist=dist, elev=elev, azim=azim)
        T += offset
        self.cameras = renderer.FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)

    def rasterization_setting(self, image_size=512, blur_radius=0.0, face_per_pixel=1):
        self.raster_setting = renderer.RasterizationSettings(image_size=image_size, blur_radius=blur_radius, faces_per_pixel=face_per_pixel, cull_backfaces=True)

    def light_setting(self, directions=[[1.0, 1.0, 1.0]]):
        self.lights = renderer.DirectionalLights(direction=directions, device=self.device)

    