loaded = load('vessel_unet.mat');
segNet = loaded.net;

testImg = imread('data/aptos2019/images/some_image.png');
testImg = imresize(testImg, [512 512]);

probMap = predict(segNet, testImg);
disp(size(probMap))

vesselMask = probMap(:,:,2) > 0.5;
imshow(labeloverlay(testImg, vesselMask));