import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer


class SlideMAEV3(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        n_slides,
        dtypes,
        dtype_to_shape_args,
        slide_dtype_order,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
    ):
        super().__init__()

        self.dtypes = dtypes
        self.dtype_to_shape_args = dtype_to_shape_args
        self.slide_dtype_order = slide_dtype_order

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.num_patches, self.encoder_dim = encoder.pos_embedding.shape[-2:]

        self.n_slides = n_slides
        self.slide_embedding = nn.Embedding(self.n_slides, self.encoder_dim)
        self.dtype_embedding = nn.Embedding(len(dtypes), self.encoder_dim)

        self.dtype_to_patch = nn.ModuleList()
        self.dtype_to_patch_emb = nn.ModuleList()
        self.dtype_to_decoder_pixels = nn.ModuleList()

        for dtype in dtypes:
            args = dtype_to_shape_args[dtype]
            n_pixels = args.patch_size * args.patch_size * args.n_channels
            self.dtype_to_patch.append(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=args.patch_size, p2=args.patch_size)
            )
            self.dtype_to_patch_emb.append(
                nn.Sequential(
                    nn.LayerNorm(n_pixels),
                    nn.Linear(n_pixels, self.encoder_dim),
                    nn.LayerNorm(self.encoder_dim),
                )
            )
            self.dtype_to_decoder_pixels.append(
                nn.Linear(decoder_dim, n_pixels)
            )

        # self.dtype_to_pixel_values_per_patch = {
        #     dtype: args.patch_size * args.patch_size * args.n_channels
        #     for dtype, args in dtype_to_shape_args.items()
        # }
        # self.to_patch = encoder.to_patch_embedding[0]
        # self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        # pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(self.encoder_dim, decoder_dim) if self.encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(self.num_patches + 1, decoder_dim)
        # self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def encode(self, stacked_imgs, slides, dtypes):
        """
        img - list of (b c h w)
        slides - list of (b)
        dtypes - list of (b)
        """
        device = stacked_imgs[0].device
        num_patches, encoder_dim = self.encoder.pos_embedding.shape[-2:]

        assert len(stacked_imgs) == len(dtypes)

        tokens = []
        flattened_slides = []
        flattened_dtypes = []
        for i in range(len(self.dtypes)):
            img, slide, dtype = stacked_imgs[i], slides[i], dtypes[i]
            patches = self.dtype_to_patch[i](img)
            tokens.append(self.dtype_to_patch_emb[i](patches))
            flattened_slides.append(slide)
            flattened_dtypes.append(dtype)
        tokens = torch.concat(tokens)
        flattened_slides = torch.concat(flattened_slides)
        flattened_dtypes = torch.concat(flattened_dtypes)

        # add slide/dtype embedding
        slide_tokens = self.slide_embedding(flattened_slides) # b d
        slide_tokens += self.dtype_embedding(flattened_dtypes)
        slide_tokens = rearrange(slide_tokens, 'b d -> b 1 d')
        slide_tokens += self.encoder.pos_embedding[:, :1]

        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # add slide/dtype token
        tokens = torch.cat((slide_tokens, tokens), dim=1)

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        return encoded_tokens, flattened_slides, flattened_dtypes
    
    def decode(self, encoded_tokens):
        device = encoded_tokens.device
        
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        decoder_tokens += self.decoder_pos_emb(torch.arange(decoder_tokens.shape[1], device=device))

        decoded_tokens = self.decoder(decoder_tokens)

        return decoded_tokens
    
    def to_pixel_values(self, decoded_tokens, flattened_dtypes):
        pixels = []
        for i in range(len(self.dtypes)):
            idxs = torch.argwhere(flattened_dtypes==i).flatten()
            decoded_subset = decoded_tokens[idxs]
            pixels_subset = self.dtype_to_decoder_pixels[i](decoded_subset)
            pixels.append(pixels_subset)
        
        return pixels


    def forward(self, imgs):
        """
        imgs - list of (b c h w), should be len(self.dtypes)
        """
        device = imgs[0].device
        assert len(self.dtypes) == len(imgs)

        bs = imgs[0].shape[0]
        slides, dtypes = [], []
        for i in range(len(self.dtypes)):
            unrolled_img = rearrange(imgs[i], 'b n c h w -> (b n) c h w')
            imgs[i] = unrolled_img
            dtypes.append(torch.full((unrolled_img.shape[0],), i, device=device))
            slides.append(repeat(
                torch.arange(6, device=device)[self.slide_dtype_order==i],
                'n -> (b n)', b=bs
            ))

        encoded_tokens, flattened_slides, flattened_dtypes = self.encode(imgs, slides, dtypes)
        
        decoded_tokens = self.decode(encoded_tokens)

        pixels = self.to_pixel_values(decoded_tokens[:, 1:], flattened_dtypes)
        recon_loss = 0
        for i, (pred_pixel_values, img) in enumerate(zip(pixels, imgs)):
            recon_loss += F.mse_loss(pred_pixel_values, self.dtype_to_patch[i](img))

        return recon_loss, pixels


class SlideMAEV2(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        n_slides,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
    ):
        super().__init__()

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.n_slides = n_slides
        self.slide_embedding = nn.Embedding(self.n_slides, encoder_dim)

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches + 1, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def encode(self, img, slides):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)

        # add slide emb
        slide_tokens = self.slide_embedding(slides) # b d
        slide_tokens = rearrange(slide_tokens, 'b d -> b 1 d')
        slide_tokens += self.encoder.pos_embedding[:, :1]

        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # add slide token

        tokens = torch.cat((slide_tokens, tokens), dim=1)

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        return encoded_tokens
    
    def decode(self, encoded_tokens):
        device = encoded_tokens.device
        
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        decoder_tokens += self.decoder_pos_emb(torch.arange(decoder_tokens.shape[1], device=device))

        decoded_tokens = self.decoder(decoder_tokens)

        return decoded_tokens


    def forward(self, img, slides):

        encoded_tokens = self.encode(img, slides)
        
        decoded_tokens = self.decode(encoded_tokens)

        pred_pixel_values = self.to_pixels(decoded_tokens[:, 1:])

        recon_loss = F.mse_loss(pred_pixel_values, self.to_patch(img))

        return recon_loss, pred_pixel_values


class SlideMAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        n_slides,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.n_slides = n_slides
        self.slide_embedding = nn.Embedding(self.n_slides, encoder_dim)

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches + 1, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def encode_all(self, img, slides):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)

        # add slide emb
        slide_tokens = self.slide_embedding(slides) # b d
        slide_tokens = rearrange(slide_tokens, 'b d -> b 1 d')
        slide_tokens += self.encoder.pos_embedding[:, :1]

        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        # num_masked = int(self.masking_ratio * num_patches)
        # rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        # masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # # get the unmasked tokens to be encoded

        # batch_range = torch.arange(batch, device = device)[:, None]
        # tokens = tokens[batch_range, unmasked_indices]
        tokens = torch.cat((slide_tokens, tokens), dim=1)

        # # get the patches to be masked for the final reconstruction loss

        # masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        return encoded_tokens
    
    def decode_all(self, encoded_tokens):
        device = encoded_tokens.device
        
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        decoder_tokens += self.decoder_pos_emb(torch.arange(decoder_tokens.shape[1], device=device))

        # decoder_slide, decoder_tokens = decoder_tokens[:, :1], decoder_tokens[:, 1:]

        # decoder_slide += self.decoder_pos_emb(torch.tensor([0], device=img.device))

        # # reapply decoder position embedding to unmasked tokens

        # unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices + 1)

        # # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices + 1)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        # decoder_tokens = torch.zeros(batch, num_patches + 1, self.decoder_dim, device=device)
        # decoder_tokens[:, :1] = decoder_slide
        # decoder_tokens[batch_range, unmasked_indices + 1] = unmasked_decoder_tokens
        # decoder_tokens[batch_range, masked_indices + 1] = mask_tokens

        decoded_tokens = self.decoder(decoder_tokens)
        pred_pixel_values = self.to_pixels(decoded_tokens[:, 1:])

        return decoded_tokens, pred_pixel_values

        # # splice out the mask tokens and project to pixel values

        # mask_tokens = decoded_tokens[batch_range, masked_indices + 1]

        # pred_pixel_values = self.to_pixels(mask_tokens)

        # # calculate reconstruction loss

        # recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        # return recon_loss, decoded_tokens


    def forward(self, img, slides):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)

        # add slide emb
        slide_tokens = self.slide_embedding(slides) # b d
        slide_tokens = rearrange(slide_tokens, 'b d -> b 1 d')
        slide_tokens += self.encoder.pos_embedding[:, :1]

        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        tokens = torch.cat((slide_tokens, tokens), dim=1)

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        decoder_slide, decoder_tokens = decoder_tokens[:, :1], decoder_tokens[:, 1:]

        decoder_slide += self.decoder_pos_emb(torch.tensor([0], device=img.device))

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices + 1)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices + 1)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches + 1, self.decoder_dim, device=device)
        decoder_tokens[:, :1] = decoder_slide
        decoder_tokens[batch_range, unmasked_indices + 1] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices + 1] = mask_tokens

        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices + 1]

        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        return recon_loss, decoded_tokens
