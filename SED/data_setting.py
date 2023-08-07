# experiment parameter setting
class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = 'EX012'
    seed = 71
    epochs = 23
    cutmix_and_mixup_epochs = 18
    folds = [ 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-3
    ETA_MIN = 1e-6
    fETA_MIN = 1e-7
    WEIGHT_DECAY = 1e-6
    train_bs = 4 # 32
    valid_bs = 4 # 64
    base_model_name = "tf_efficientnet_b0_ns"
    scheduler = 'CosineAnnealingLR'
    EARLY_STOPPING = True
    DEBUG = False         # True
    EVALUATION = 'AUC'    #采用auc作为评价指标
    apex = True

    pooling = "max"
    pretrained = True
    num_classes = 152
    in_channels = 3
    target_columns = 'afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter \
                      barpet bcnher belkin1 bkbplo bknsti bkwpet blkfra blknod bongul \
                      brant brnboo brnnod brnowl brtcur bubsan buffle bulpet burpar buwtea \
                      cacgoo1 calqua cangoo canvas caster1 categr chbsan chemun chukar cintea \
                      comgal1 commyn compea comsan comwax coopet crehon dunlin elepai ercfra eurwig \
                      fragul gadwal gamqua glwgul gnwtea golphe grbher3 grefri gresca gryfra gwfgoo \
                      hawama hawcoo hawcre hawgoo hawhaw hawpet1 hoomer houfin houspa hudgod iiwi incter1 \
                      jabwar japqua kalphe kauama laugul layalb lcspet leasan leater1 lessca lesyel lobdow lotjae \
                      madpet magpet1 mallar3 masboo mauala maupar merlin mitpar moudov norcar norhar2 normoc norpin \
                      norsho nutman oahama omao osprey pagplo palila parjae pecsan peflov perfal pibgre pomjae puaioh \
                      reccar redava redjun redpha1 refboo rempar rettro ribgul rinduc rinphe rocpig rorpar rudtur ruff \
                      saffin sander semplo sheowl shtsan skylar snogoo sooshe sooter1 sopsku1 sora spodov sposan \
                      towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov'.split()
    
    scored_birds = ["akiapo", "aniani", "apapan", "barpet", "crehon", "elepai", "ercfra", "hawama", "hawcre", "hawgoo", "hawhaw", "hawpet1", "houfin", "iiwi", "jabwar", "maupar", "omao", "puaioh", "skylar", "warwhe1", "yefcan"]
    scored_id = [3, 6, 7, 9, 44, 46, 47, 60, 62, 63, 64, 65, 67, 70, 72, 90, 101, 111, 131, 141, 150]
    bird2id = {b:i for i,b in enumerate(target_columns)}
    id2bird = {i:b for i,b in enumerate(target_columns)}

    img_size = 224 # 128
    main_metric = "epoch_f1_at_03"   #？f1作为度量？

    period = 5
    n_mels = 224 # 128
    len_check = 313
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    melspectrogram_parameters = {
        "n_mels": 224, # 128,
        "fmin": 20,
        "fmax": 16000
    }

    #multiprocess_training
    num_workers = 32
    rating = None
    
    
class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = CFG.sample_rate
    duration = CFG.period
    # Melspectrogram
    n_mels = CFG.n_mels
    fmin = CFG.fmin
    fmax = CFG.fmax
