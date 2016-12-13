-- Image Denoising for images of arbitrary size (assuming > 64x64)
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
    img_path = '',         -- path to the test image
    net = '',              -- path to the image denoising network
    name = 'denoise',      -- name of the experiment and prefix of file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    fineSize = 64,         -- size of overlapping patches
    sigma = 25,            -- size of gassian blur kernel
    stepSize = 3,          -- stride size of overlapping patches
    manualSeed = 0,        -- 0 means random seed
}

-- set seed
if opt.manualSeed == 0 then
   opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- load network
assert(opt.net ~= '', 'provide a generator model')
net = util.load(opt.net, opt.gpu)
net:evaluate()

-- load image
local input = image.load(opt.img_path, opt.nc, 'float') -- in the range (0,1)
input:mul(255) -- in the range (0,255)

-- convert to grayscale image in the range (0,255)
local input_one_channel = torch.Tensor(1, input:size(2), input:size(3))
local input_one_channel = input[{{1},{},{}}] * 0.2989 + input[{{2},{},{}}] * 0.5870 + input[{{3},{},{}}] * 0.1140

-- initialization
local real_gray_image = torch.Tensor(1, input:size(2), input:size(3)):zero()
local denoised_gray_image = torch.Tensor(1, input:size(2), input:size(3)):zero()
local criterionMSE = nn.MSECriterion()
real_gray_image:copy(input_one_channel)

-- generate while gaussian noise
local noise = torch.Tensor(1, input:size(2), input:size(3))
noise:normal(0,opt.sigma)

-- corrupt image with white gaussian noise
input_one_channel:add(noise)

-- clamp pixel values out of range (0,255)
input_one_channel = torch.clamp(input_one_channel,0,255)
input_one_channel:div(255):mul(2):add(-1) -- in the range (-1,1), the range that the image denoising network is trained for

-- denoising initialization
local image_width = input_one_channel:size(2)
local image_height = input_one_channel:size(3)
local stepSize = opt.stepSize
local batchSize = 100
local input_image = torch.Tensor(batchSize, 1, opt.fineSize, opt.fineSize)
local patch_parameters = torch.Tensor(batchSize, 2)
local width_start = 1 
local width_end = 64
local height_start = 1
local height_end = 64
local patch_index = 0
local output_image = torch.Tensor(batchSize, 1, opt.fineSize, opt.fineSize)
local image2process = torch.Tensor(1, opt.fineSize, opt.fineSize)
local count_index = torch.Tensor(1, image_width, image_height):zero() -- count the number of overlapping patches for each pixel

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
    output_image = output_image:cuda()
    criterionMSE = criterionMSE:cuda()
end

function update()
	output_image = net:forward(input_image)
	for i = 1,patch_index do 
		w_start = torch.squeeze(patch_parameters[{{i},{1}}])
		h_start = torch.squeeze(patch_parameters[{{i},{2}}])
		image2process:copy(output_image[i])
    -- add denoised patch
		denoised_gray_image[{{1},{w_start,w_start+63},{h_start,h_start+63}}]:add(image2process)
		-- increase counter of overlap patches per pixel
    count_index[{{1},{w_start,w_start+63},{h_start,h_start+63}}]:add(1)
	end
	patch_index = 0
end
		
-- denoise in a sliding-window manner patch by patch (64x64)
while height_end < image_height do
  while width_end < image_width do
      patch_index = patch_index + 1
      input_image[{{patch_index},{1},{},{}}] = input_one_channel[{{1},{width_start,width_end},{height_start,height_end}}]
      patch_parameters[{{patch_index},{1}}] = width_start
      patch_parameters[{{patch_index},{2}}] = height_start
      width_start = width_start + stepSize
      width_end = width_end + stepSize
      -- denoise as a batch for efficiency
      if patch_index == batchSize then update() end
  end
  -- sliding-window gets to the horizontal boundary 
  if width_end >= image_width then
      patch_index = patch_index + 1
      input_image[{{patch_index},{1},{},{}}] = input_one_channel[{{1},{image_width-63,image_width},{height_start,height_end}}]
      patch_parameters[{{patch_index},{1}}] = image_width-63
      patch_parameters[{{patch_index},{2}}] = height_start
      height_start = height_start + stepSize
      height_end = height_end + stepSize
      width_start = 1
      width_end = 64
      -- denoise as a batch for efficiency
      if patch_index == batchSize then update() end
  end
  -- sliding-window gets to the vertical boundary
  if height_end >= image_height then
      while width_end < image_width do
	      patch_index = patch_index + 1
        input_image[{{patch_index},{1},{},{}}] = input_one_channel[{{1},{width_start,width_end},{image_height-63,image_height}}]
        patch_parameters[{{patch_index},{1}}] = width_start
        patch_parameters[{{patch_index},{2}}] = image_height-63
        width_start = width_start + stepSize
        width_end = width_end + stepSize
        -- denoise as a batch for efficiency
        if patch_index == batchSize then update() end
      end
      -- sliding-window gets to the last patch (gets to both the horizontal boundary and vertical boundary) to denoise
      if width_end >= image_width then
	      patch_index = patch_index + 1
        input_image[{{patch_index},{1},{},{}}] = input_one_channel[{{1},{image_width-63,image_width},{image_height-63,image_height}}]
	      patch_parameters[{{patch_index},{1}}] = image_width-63
        patch_parameters[{{patch_index},{2}}] = image_height-63
        update()
      end
  end 
end

-- transform back to original scale
input_one_channel:add(1):mul(0.5):mul(255)
denoised_gray_image:cdiv(count_index)
denoised_gray_image:add(1):mul(0.5):mul(255)

-- save outputs in a pretty manner
pretty_output = torch.Tensor(3, 1, real_gray_image:size(2), real_gray_image:size(3))
pretty_output[1]:copy(real_gray_image)
pretty_output[2]:copy(input_one_channel)
pretty_output[3]:copy(denoised_gray_image)
image.save(opt.name .. '.png', image.toDisplayTensor(pretty_output))

-- compute PSNR of the restored image
if opt.gpu then
   real_gray_image = real_gray_image:cuda()
   denoised_gray_image = denoised_gray_image:cuda()
end
l2loss = criterionMSE:forward(denoised_gray_image, real_gray_image)
PSNR = 10 * math.log10(255*255 / l2loss)
print("PSNR of the restored image: " ..  PSNR)
