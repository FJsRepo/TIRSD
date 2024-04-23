from ImgEnhance.SSLModelLoad import SSLModelLoad
from ImgEnhance.enhance import imgEnhance

def main():

    # ##################---Parameter settings---####################
    epoch = 400
    # Original image path
    rawFilePathTrain = './MAR-DCT-Original/train_images'
    rawFilePathTest = './MAR-DCT-Original/val_images'

    # Enhanced image save path
    enhancedSavedPathTrain = './MAR-DCT/train_images/'
    enhancedSavedPathTest = './MAR-DCT/val_images/'
    # ################################################################
    print('Loading SSL model...')
    SSLModel = SSLModelLoad(epoch)
    print('Start image enhance...')
    imgEnhance(SSLModel, rawFilePathTrain, rawFilePathTest, enhancedSavedPathTrain, enhancedSavedPathTest)

if __name__ == '__main__':
    main()
