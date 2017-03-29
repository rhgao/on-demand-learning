-- code derived from DCGAN and Context Encoder
-- DCGAN: https://github.com/soumith/dcgan.torch
-- Context Encoder: https://github.com/pathak22/context-encoder
-- Ruohan Gao. All rights reserved.
-- Dec. 2016

require 'image'
require 'nn'
local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'

util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 100,       -- number of samples to produce
    net = '',              -- path to the encoder-decoder network
    name = 'denoise',      -- name of the experiment and prefix of file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    display = 1,           -- Display image: 0 = false, 1 = true
    loadSize = 96,         -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
    fineSize = 64,         -- size of random crops
    sigma = 25,            -- noise sigma
    nThreads = 1,          -- # of data loading threads to use
    manualSeed = 0,        -- 0 means random seed
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- load image deblurrer
assert(opt.net ~= '', 'provide a generator model')
net = util.load(opt.net, opt.gpu)
net:evaluate()

-- initialize variables
local input_image = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local gray_image = torch.Tensor(opt.batchSize,1,opt.fineSize,opt.fineSize)
local real_frame = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local fake_frame = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    if pcall(require, 'cudnn') then
        print('Using CUDNN !')
        require 'cudnn'
        net = util.cudnn(net)
    end
    net:cuda()
    input_image = input_image:cuda()
else
   net:float()
end
print(net)

-- load data
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
local real_image = data:getBatch()
print('Loaded Image Block: ', real_image:size(1)..' x '..real_image:size(2) ..' x '..real_image:size(3)..' x '..real_image:size(4))

-- construct corrupted images
real_image:add(1):mul(0.5):mul(255)
real_frame[{{},{},{},{}}] = real_image[{{},{1},{},{}}] * 0.2989 + real_image[{{},{1},{},{}}] * 0.5870 + real_image[{{},{1},{},{}}] * 0.1140
gray_image:copy(real_frame)
real_frame:div(255):mul(2):add(-1)
for i = 1, opt.batchSize do
    local noise = torch.Tensor(1, opt.fineSize, opt.fineSize)
    noise:normal(0,opt.sigma)
    gray_image[{{i},{},{},{}}]:add(noise)
end
gray_image = torch.clamp(gray_image, 0, 255)
gray_image:div(255):mul(2):add(-1)
input_image:copy(gray_image)

-- run image denoiser
local fake = net:forward(input_image)
fake_frame[{{},{},{},{}}]:copy(fake)

-- re-transform scale back to normal
real_frame:add(1):mul(0.5)
fake_frame:add(1):mul(0.5)
gray_image:add(1):mul(0.5)

if opt.display then
    disp = require 'display'
    disp.image(real_frame, {win=1000, title=opt.name .. '-original image'})
    disp.image(gray_image, {win=1001, title=opt.name .. '-corrupted image'})
    disp.image(fake_frame, {win=1002, title=opt.name .. '-restored image'})
    print('Displayed image in browser !')
end

-- save outputs in a pretty manner
pretty_output = torch.Tensor(3*opt.batchSize, 1, opt.fineSize, opt.fineSize)
for i=1,opt.batchSize do
    pretty_output[3*i-2]:copy(real_frame[i])
    pretty_output[3*i-1]:copy(gray_image[i])
    pretty_output[3*i]:copy(fake_frame[i])
end
image.save(opt.name .. '.png', image.toDisplayTensor(pretty_output))
print('Saved predictions to: ./', opt.name .. '.png')