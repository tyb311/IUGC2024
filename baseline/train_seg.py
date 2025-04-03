from utils.dataset_segmentation import *
from utils.augmentation import *
import random
from utils.criterion import MyCriterion
from torch.utils.data import DataLoader
from utils.metric import DSC
from model import U_Net, IUGCNet


def get_model(path_ckpt = r'deeplabv3.pt'):
    # model = U_Net(mode="seg")
    model = IUGCNet()
    return model


def get_dataloader(img_size=512, batch_size=2):
    tf_train = JointTransform2D(img_size=img_size, low_img_size=128, ori_size=img_size, crop=None, p_flip=0.5,
                                p_rota=0.5, p_scale=0.0, p_gaussn=0.0, p_contr=0.0, p_gama=0.0, p_distor=0.0,
                                color_jitter_params=None,
                                long_mask=True)
    tf_val = JointTransform2D(img_size=img_size, low_img_size=128, ori_size=img_size, crop=None, p_flip=0.0,
                              p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0, p_gama=0.0, p_distor=0.0,
                              color_jitter_params=None,
                              long_mask=True)

    train_set = DatasetSegmentation("./train/pos", tf_train)
    val_set = DatasetSegmentation("./val/pos", tf_val)
    dataset_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_val = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return dataset_train, dataset_val


def save_training_process(train_losses, val_losses, start_val_epoch):
    with open("./process.txt", "w") as file:
        num_epochs = len(train_losses)
        for i in range(num_epochs):
            file.write(f"Epoch {i} Train Loss: {train_losses[i]}")
            if (i + 1) >= start_val_epoch:
                index = i - (start_val_epoch - 1)
                file.write(f"Epoch {i} Val DSC: {val_losses[index]}")
            file.write("\n")


def main():
    # ========================== set random seed ===========================#
    seed_value = 2024  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    # ========================== set hyper parameters =========================#
    BATCH_SIZE = 4
    NUM_EPOCHS = 80
    BASE_LEARNING_RATE = 0.0001
    IMG_SIZE = 512
    START_VAL_EPOCH = 40

    # ========================== get model, dataloader, optimizer and so on =========================#
    model = get_model()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LEARNING_RATE,
                                  betas=(0.9, 0.999), weight_decay=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # model.load(f"xxxx.pth")

    dataset_train, dataset_val = get_dataloader(IMG_SIZE, BATCH_SIZE)

    criterion = MyCriterion()  # combined loss function
    evalue = DSC()  # metric to find best model
    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_val_score = 0
    train_losses = []
    val_scores = []

    # ========================== training =============================#
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, sample in enumerate(dataset_train):
            print(f"Epoch[{epoch + 1}/{NUM_EPOCHS}] | Batch {batch_idx}: ", end="")
            batch_train_loss = []
            imgs = sample['image'].to(dtype=torch.float32, device=device)  # (BS, 3, 512, 512)
            masks = sample['label'].to(device=device).squeeze(1)  # (BS,512,512)   0: background 1:ps 2:fh

            # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            preds = model(imgs, mode="seg")
            train_loss = criterion(preds, masks)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_train_loss.append(train_loss.detach().cpu().numpy())

            print(f"Batch {batch_idx}:", end="")
        train_losses.append(np.mean(batch_train_loss))

        # ========================= validation =======================#
        if (epoch + 1) >= START_VAL_EPOCH:
            model.eval()
            val_score_ls = []
            with torch.no_grad():
                for batch_idx, sample in enumerate(dataset_val):
                    imgs = sample['image'].to(dtype=torch.float32, device=device)  # (BS,3,512,512)
                    masks = sample['label'].to(device=device).squeeze(1)  # (BS,512,512)
                    preds = model(imgs, mode="seg")
                    val_score_one_batch = evalue(preds, masks)
                    val_score_ls.append(val_score_one_batch.detach().cpu().numpy())
                val_score = np.mean(val_score_ls)
                val_scores.append(val_score)

                # ================  SAVING ================#
                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), f"./checkpoints/epoch{epoch}_val_score_{val_score:.6f}.pth")

    save_training_process(train_losses, val_scores, START_VAL_EPOCH)


if __name__ == '__main__':
    main()
