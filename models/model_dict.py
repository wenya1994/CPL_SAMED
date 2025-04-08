from models.SAMED.build_sam_us import samus_model_registry


def get_model(modelname="SAMED", args=None, opt=None):
    model = samus_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    return model
