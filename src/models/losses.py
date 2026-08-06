import torch
import torch.nn as nn
class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.MSELoss()
 
    def _get_target_tensor(self, prediction, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)
 
    def forward(self, prediction, target_is_real):
        target_tensor = self._get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
def cycle_consistency_loss(real_image, reconstructed_image):
    return nn.functional.l1_loss(reconstructed_image, real_image)
def identity_loss(real_image, same_domain_output):
    return nn.functional.l1_loss(same_domain_output, real_image)
class CycleGANLosses:
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5, device="cpu"):
        self.gan_loss = GANLoss().to(device)
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity * lambda_cycle
    def compute_generator_loss(
        self,
        D_face, D_anime,
        real_face, real_anime,
        fake_anime, fake_face,          
        reconstructed_face, reconstructed_anime,  
        identity_face, identity_anime,  
    ):
        loss_adv_face2anime = self.gan_loss(D_anime(fake_anime), target_is_real=True)
        loss_adv_anime2face = self.gan_loss(D_face(fake_face), target_is_real=True)
        loss_cycle_face = cycle_consistency_loss(real_face, reconstructed_face)
        loss_cycle_anime = cycle_consistency_loss(real_anime, reconstructed_anime)
        loss_cycle = loss_cycle_face + loss_cycle_anime
        loss_id_face = identity_loss(real_face, identity_face)
        loss_id_anime = identity_loss(real_anime, identity_anime)
        loss_identity = loss_id_face + loss_id_anime
        total = (
            loss_adv_face2anime + loss_adv_anime2face
            + self.lambda_cycle * loss_cycle
            + self.lambda_identity * loss_identity
        )
        return total, {
            "adv_face2anime": loss_adv_face2anime.item(),
            "adv_anime2face": loss_adv_anime2face.item(),
            "cycle": loss_cycle.item(),
            "identity": loss_identity.item(),
            "total": total.item(),
        }
    def compute_discriminator_loss(self, D, real, fake):
        """
        Standard LSGAN discriminator loss: real images should score close to 1,
        fake (detached, so no generator gradients flow here) should score close to 0.
        Call once per discriminator (D_face, D_anime), each with its own real/fake pair.
        """
        loss_real = self.gan_loss(D(real), target_is_real=True)
        loss_fake = self.gan_loss(D(fake.detach()), target_is_real=False)
        return (loss_real + loss_fake) * 0.5
if __name__ == "__main__":
    import torch
    from generator import ResnetGenerator
    from discriminator import PatchDiscriminator
    device = "cpu"
    G_face2anime = ResnetGenerator(n_residual_blocks=9).to(device)
    G_anime2face = ResnetGenerator(n_residual_blocks=9).to(device)
    D_face = PatchDiscriminator().to(device)
    D_anime = PatchDiscriminator().to(device)
    real_face = torch.randn(1, 3, 256, 256)
    real_anime = torch.randn(1, 3, 256, 256)
    fake_anime = G_face2anime(real_face)
    fake_face = G_anime2face(real_anime)
    reconstructed_face = G_anime2face(fake_anime)
    reconstructed_anime = G_face2anime(fake_face)
    identity_face = G_anime2face(real_face)
    identity_anime = G_face2anime(real_anime)
    losses = CycleGANLosses(device=device)
    total_g_loss, log = losses.compute_generator_loss(
        D_face, D_anime,
        real_face, real_anime,
        fake_anime, fake_face,
        reconstructed_face, reconstructed_anime,
        identity_face, identity_anime,
    )
    print("Generator loss breakdown:", log)
    d_face_loss = losses.compute_discriminator_loss(D_face, real_face, fake_face)
    d_anime_loss = losses.compute_discriminator_loss(D_anime, real_anime, fake_anime)
    print("D_face loss:", d_face_loss.item(), "| D_anime loss:", d_anime_loss.item())
 