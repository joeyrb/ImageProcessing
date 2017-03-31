--[[
Author: Joey Brown
Class: CSC 442, Digitial Image Processing
Asgmnt: PA #2
Description: 

Acknowledgements:
  * 
--]]

require "ip"
local viz = require "visual"
local il = require "il"

local imgs = {...}
for i, fname in ipairs(imgs) do loadImage(fname) end


-- Edge detection results should be mapped to grayscale values.--
--[[ 
Extra credit possibilities:
1. Gaussian smoothing filter (prompt user for σ)
2. Marr-Hildreth edge operator: Laplacian of Gaussian or Difference of Gaussians (prompt user
for σ), followed by detection of zero crossings
3. Canny edge operator (will cover in lecture)
4. Implement filters using separability (significant speedup for large filter sizes)
5. Handle image borders by reflection (instead of ignoring them)
--]]

---------------
-- functions --
---------------

-- MISC --
-- binary threshold --
local function bThreshold ( img, input )
  img = il.RGB2YIQ( img )

  for r,c in img:pixels() do
    if ( img:at(r,c).y >= input) then
      img:at(r,c).r = 255
      img:at(r,c).g = 255
      img:at(r,c).b = 255
    else
      img:at(r,c).r = 0
      img:at(r,c).g = 0
      img:at(r,c).b = 0
    end
    
  end
  return img
end
-----------------------------------


-- NEIGHBORHOOD OPERATIONS --
-- 3x3 Smoothing Filter --
--[[ Mask:
       1, 2, 1,
       2, 4, 2,
       1, 2, 1
       
       Multiply by 1/16 after
  --]]
local function smooth3x3( img )
  -- convert to grayscale to measure intensity
  img = il.RGB2IHS( img )
  local res = img:clone() -- clone image so changed pixels dont affect neighbors
  
  -- for every pixel in image (not including the borders)
  for r,c in img:pixels(1) do
    -- sum intensities 
    local sum = 0
    
    sum = sum + img:at(r-1, c-1).ihs[0] * 1
    sum = sum + img:at(r-1, c  ).ihs[0] * 2
    sum = sum + img:at(r-1, c+1).ihs[0] * 1
    
    sum = sum + img:at(r, c-1).ihs[0] * 2
    sum = sum + img:at(r, c  ).ihs[0] * 4
    sum = sum + img:at(r, c+1).ihs[0] * 2
    
    sum = sum + img:at(r+1, c-1).ihs[0] * 1
    sum = sum + img:at(r+1, c  ).ihs[0] * 2
    sum = sum + img:at(r+1, c+1).ihs[0] * 1
    
    -- divide center pixel by 16 and store in res
    res:at(r,c).ihs[0] = sum / 16
  end
  
  -- convert img back to RGB
  return il.IHS2RGB( res )
end
-----------------------------------

-- 3x3 Sharpening Filter --
--[[ Mask:
      0, -1,  0,
     -1,  5, -1,
      0, -1,  0
--]]
local function sharpen3x3( img )
  img = il.RGB2IHS( img )
  local res = img:clone()
  -- write result of mask applied to img to res
  -- (same as smoothing with a different mask)
  
  for r,c in img:pixels(1) do
    local sum = 0
    
    --sum = sum + img:at(r-1, c-1).ihs[0] * 0
    sum = sum + img:at(r-1, c  ).ihs[0] * -1
    --sum = sum + img:at(r-1, c+1).ihs[0] * 0
    
    sum = sum + img:at(r, c-1).ihs[0] * -1
    sum = sum + img:at(r, c  ).ihs[0] * 5
    sum = sum + img:at(r, c+1).ihs[0] * -1
    
    --sum = sum + img:at(r+1, c-1).ihs[0] * 0
    sum = sum + img:at(r+1, c  ).ihs[0] * -1
    --sum = sum + img:at(r+1, c+1).ihs[0] * 0
    
    -- check for out of bounds pixels then add absolute value of sum
    if sum <= 255 then
      res:at(r,c).ihs[0] = math.abs( sum )
    elseif sum > 255 then
      res:at(r,c).ihs[0] = 255
    elseif sum < 0 then
      res:at(r,c).ihs[0] = 0
    end
  end
  
  return il.IHS2RGB( res )
end
-----------------------------------

-- Plus Shaped Median Filter --
--[[
    0, 1, 0,
    1, 1, 1,
    0, 1, 0
--]]
local function plusShape( img )
 local nrows, ncols = img.height, img.width 
  img = il.RGB2YIQ( img )
  local res = img:clone()
  
  for r, c in img:pixels(1) do
    local arr = {}
    local sum = 0
    
    -- Insert only the pixels in the plus shape of the mask
    --  since the zeros are used to ignore the corners.
    table.insert(arr, img:at(r-1, c).y)
    table.insert(arr, img:at(r, c-1).y)
    table.insert(arr, img:at(r, c  ).y)
    table.insert(arr, img:at(r, c+1).y)
    table.insert(arr, img:at(r+1, c).y)
    
    table.sort(arr)
    
    -- set the center pixel to the median value in the table
    res:at(r,c).y = arr[3]
    
  end

  
  return il.YIQ2RGB( res )
end
-----------------------------------

-- Out-of-Range Noise Cleaning Filter --
local function outOfRngNoiseClean( img, thrsh )
--[[ User gives threshold (thrsh) that offets the center pixel 
      by the value of thrsh in the positive and negative direction;
      this is your range. Then, the average of the neighborhood is 
      computed and checks to see if it is within the range of the
      threshold. If the neighborhood average is within the range, then
      nothing is changed. Otherwise, change the center pixel intensity 
      value to the neighborhood average. In this function, the range
      is inclusive. --]]
   
  img = il.RGB2YIQ( img )
  local res = img:clone()
   
   for r, c in img:pixels(1) do
    local sum = 0
    local range = {}
    local min = img:at(r,c).y - thrsh -- range minimum
    if min < 0 then
      min = 0
    end
    local max = img:at(r,c).y + thrsh -- range maximum
    if max > 255 then
      max = 255
    end
    
    -- for every pixel, iterate through the filter mask
    for i = -1, 1 do
      for j = -1, 1 do
        sum = sum + img:at(r+i, c+j).y
      end
    end
    
    sum = (sum / (9*255))*255 -- neighborhood average
    
    -- if the neighborhood average is out of range then change 
    --    the center pixel to the average. 
    if sum < min or sum > max then
      res:at(r,c).y = sum
    end
    
  end
  
  return il.YIQ2RGB( res )
end
-----------------------------------

-- n x n neighborhood mean filter --
local function mean( img, n )
-- change the middle pixel to the neighborhood average
  local nrows, ncols = img.height, img.width
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local border = math.floor( n / 2 )
  
  -- check if n is odd
  if math.fmod(n, 2) == 0 then 
    return il.YIQ2RGB( img )
  end
  
  -- check if n < size(img)
  if n > nrows or  n > ncols then
    return il.YIQ2RGB( img )
  end
  
  local m = (n-1)/2
  -- range: m <= x <= rows-m, m <= y <= col-m
  for r,c in img:pixels(border) do
    local sum = 0
    
    -- for every pixel, iterate through the filter mask
    for i = (r-m), (r+m) do
      for j = (c-m),(c+m) do
        sum = sum + img:at(i, j).y
      end
    end
    
    sum = (sum / (n*n*255))*255
    
    -- check for out of bounds intensity values
    if sum < 255 and sum > 0 then
      res:at(r,c).yiq[0] = sum
    elseif sum < 0 then
      res:at(r,c).yiq[0] = 0
    else
      res:at(r,c).yiq[0] = 255
    end
      
      
  end
    
  return il.YIQ2RGB( res )
end
-----------------------------------

-- n x n neighborhood median filter --
local function median( img, n )
--[[ Add all of the neighborhood values to an array (or table) 
      of size(n). Set middle pixel of filter to the median of
      the array.
 --]]
  
  local nrows, ncols = img.height, img.width 
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local border = math.floor( n / 2 )
  
  -- check if n is odd
  if math.fmod(n, 2) == 0 then 
    return il.YIQ2RGB( img )
  end
  
  -- check if n < size(img)
  if n > nrows or  n > ncols then
    return il.YIQ2RGB( img )
  end
  
  local m = (n-1)/2
  for r, c in img:pixels(border) do
    local arr = {}
    
    -- fill the mask with the image values
    for i = (r-m), (r+m) do
      for j = (c-m), (c+m) do
        table.insert(arr, img:at(i,j).y)
      end
    end
    
    -- sort the table
    table.sort(arr)
    
    -- set the center pixel to the median value in the table
    local center = math.floor((n*n)/2)+1
    res:at(r,c).y = arr[center]
    
  end

  
  return il.YIQ2RGB( res )
end
-----------------------------------

-- n x n neighborhood minimum filter --
local function min( img, n )
--[[ Set the middle pixel of the filter to the smallest
      value in the neighborhood.
--]]
  local nrows, ncols = img.height, img.width 
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local arr = {}
  local border = math.floor( n / 2 )
  
  -- check if n is odd
  if math.fmod(n, 2) == 0 then 
    return il.YIQ2RGB( img )
  end
  
  -- check if n < size(img)
  if n > nrows or  n > ncols then
    return il.YIQ2RGB( img )
  end
  
  local m = (n-1)/2
  for r, c in img:pixels(border) do
    local arr = {}
    
    -- fill the mask with the image values
    for i = (r-m), (r+m) do
      for j = (c-m), (c+m) do
        table.insert(arr, img:at(i,j).y)
      end
    end
    
    -- sort the table
    table.sort(arr)
    
    -- set the center pixel to the median value in the table
    res:at(r,c).y = arr[1]
    
  end

  
  return il.YIQ2RGB( res )
end
-----------------------------------

-- n x n neighborhood maximum filter --
local function max( img, n )
--[[ Set the middle pixel of the filter to the largest
      value in the neighborhood.
--]]
  local nrows, ncols = img.height, img.width 
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local arr = {}
  local border = math.floor( n / 2 )
  
  -- check if n is odd
  if math.fmod(n, 2) == 0 then 
    return il.YIQ2RGB( img )
  end
  
  -- check if n < size(img)
  if n > nrows or  n > ncols then
    return il.YIQ2RGB( img )
  end
  
  local m = (n-1)/2
  for r, c in img:pixels(border) do
    local arr = {}
    
    -- fill the mask with the image values
    for i = (r-m), (r+m) do
      for j = (c-m), (c+m) do
        table.insert(arr, img:at(i,j).y)
      end
    end
    
    -- sort the table
    table.sort(arr)
    
    -- set the center pixel to the median value in the table
    res:at(r,c).y = arr[table.maxn(arr)]
    
  end

  
  return il.YIQ2RGB( res )
end
-----------------------------------

-- n x n neighborhood range filter -- 
local function range( img, n )
-- change center pixel to the range of the neighborhood
  local nrows, ncols = img.height, img.width
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local border = math.floor( n / 2 )
  
  
  -- check if n is odd
  if math.fmod(n, 2) == 0 then 
    return il.YIQ2RGB( img )
  end
  
  -- check if n < size(img)
  if n > nrows or  n > ncols then
    return il.YIQ2RGB( img )
  end
  
  local m = (n-1)/2
  for r, c in img:pixels(border) do
    local sum = 0
    local min = 255
    local max = 0
  
    -- for every pixel, iterate through the filter 
    --  to find the min and max of the neighborhood
    for i = (r-m), (r+m) do
      for j = (c-m),(c+m) do
        -- check for min
        if img:at(i,j).y < min then
          min = img:at(i,j).y
        end
        
        -- check for max
        if img:at(i,j).y > max then
          max = img:at(i,j).y
        end
      end
    end
    
    -- compute the range and check bounds
    local range = max - min
    if range < 0 then
      range = 0
    elseif range > 255 then
      range = 255
    end
  
    -- set the center pixel to the value of the range
    res:at(r,c).y = range
    
  end
  
  return il.YIQ2RGB( res )
end
-----------------------------------

-- n x n neighborhood standard deviation filter --**********FIX**********
local function stddev( img, n )
-- Change the center pixel to the standard deviation of the neighborhood
  -- calculate mean
  local nrows, ncols = img.height, img.width
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local border = math.floor( n / 2 )
  
  -- check if n is odd
  if math.fmod(n, 2) == 0 then 
    return il.YIQ2RGB( img )
  end
  
  -- check if n < size(img)
  if n > nrows or  n > ncols then
    return il.YIQ2RGB( img )
  end
  
  local m = (n-1)/2
  for r,c in img:pixels(border) do
    local sum = 0
    local arr = {}
    
    -- for every pixel, iterate through the filter mask
    for i = (r-m), (r+m) do
      for j = (c-m),(c+m) do
        sum = sum + img:at(i, j).y
        table.insert(arr, img:at(i,j).y)
      end
    end
    
    local mean = (sum / (n*n*255))*255

    -- check for out of bounds intensity values
    if mean < 0 then
      mean = 0
    elseif mean > 255 then
      mean = 255
    end
    
    -- for each number, subtract the mean and square the result
    for a = 1, n do
      local result = arr[a] - mean
      arr[a] = result^2
    end
  
    -- find the mean of squared differences in the array
    local newMean = 0
    for b = 1, n do
      newMean = newMean + arr[b]
    end
    newMean = (newMean / (n*n*255))*255
    
    -- Finally, calculate the square root of the mean of the differences and
    --  set it to the center pixel
    res:at(r,c).y = math.sqrt(newMean)
    
  end
    
  return il.YIQ2RGB( res )
end
-----------------------------------

--[[
Sobel edge direction values need to be scaled to the range 0-255 for display purposes, with
intensities ranging from black at 0 degrees to white at 360 degrees. The atan2 documentation indicates that
this function returns values in the range -pi to +pi radians (-180 to +180). As discussed in
class, adding 2 pi to negative values, -pi to 0 inclusive, generates angles in the range 0 to 2 pi. This range
can easily be rescaled to 0 to 255 for display purposes.
--]]--[[--]]

--[[
Sobel algorithm (1st draft):
  - Process pixels with pix = math.atan2()
  - Add 2pi to pix <= -pi
  - Convert pix back to [0,255]
--]]

--[[ Sobel Mask:
      -1, 0, 1
      -2, 0, 2,
      -1, 0, 2
      
      -1, -2, -1,
       0,  0,  0,
      -1, -2, -1
--]]
-- Sobel edge magnitude --
local function sobelEdgeMag( img )
  img = il.RGB2YIQ( img )
  local res = img:clone()

  -- multiply neighborhood by the mask and store in Gx and Gy
  for r, c in img:pixels(1) do
    local Gx, Gy, G = 0, 0, 0
      
      -- Gx (columns)
      Gx = Gx + img:at(r-1, c-1).y * ( 1 )
      Gx = Gx + img:at(r  , c-1).y * ( 2 )
      Gx = Gx + img:at(r+1, c-1).y * ( 1 )
      
      Gx = Gx + img:at(r-1, c+1).y * (-1 )
      Gx = Gx + img:at(r,   c+1).y * (-2 )
      Gx = Gx + img:at(r+1, c+1).y * (-1 )
    
      -- Gy (rows)
      Gy = Gy + img:at(r-1, c-1).y * ( 1 )
      Gy = Gy + img:at(r-1, c  ).y * ( 2 )
      Gy = Gy + img:at(r-1, c+1).y * ( 1 )
      
      Gy = Gy + img:at(r+1, c-1).y * (-1 )
      Gy = Gy + img:at(r+1, c  ).y * (-2 )
      Gy = Gy + img:at(r+1, c+1).y * (-1 )
      
    -- Store the square root of the sum of the squares in G (Magnitude)
    G = math.floor(math.sqrt((Gx*Gx)+(Gy*Gy)))
    
    -- Clip
    if G > 255 then
      G = 255
    elseif G < 0 then
      G = 0
    end
    
    -- delete colors I and Q channel = 128
    res:at(r,c).y = G
    res:at(r,c).yiq[1] = 128
    res:at(r,c).yiq[2] = 128
  end
  
  return il.YIQ2RGB( res )
end

-- Sobel edge direction --
local function sobelEdgeDir( img )
  img = il.RGB2YIQ( img )
  local res = img:clone()

  -- multiply neighborhood by the mask and store in Gx and Gy
  for r, c in img:pixels(1) do
    local Gx, Gy, G = 0, 0, 0
      
      -- Gx (columns)
      Gx = Gx + img:at(r-1, c-1).y * (-1 )
      Gx = Gx + img:at(r  , c-1).y * (-2 )
      Gx = Gx + img:at(r+1, c-1).y * (-1 )
      
      Gx = Gx + img:at(r-1, c+1).y * ( 1 )
      Gx = Gx + img:at(r,   c+1).y * ( 2 )
      Gx = Gx + img:at(r+1, c+1).y * ( 1 )
    
      -- Gy (rows)
      Gy = Gy + img:at(r+1, c-1).y * (-1 )
      Gy = Gy + img:at(r+1, c  ).y * (-2 )
      Gy = Gy + img:at(r+1, c+1).y * (-1 )
      
      Gy = Gy + img:at(r-1, c-1).y * ( 1 )
      Gy = Gy + img:at(r-1, c  ).y * ( 2 )
      Gy = Gy + img:at(r-1, c+1).y * ( 1 )
      
     
    -- Get gradient direction
    local theta = 0
    theta = math.atan2(Gy, Gx)
    
    -- Add 2pi to negative theta values
    if theta < 0 then
      theta = (theta + (2*math.pi))
    end
    
    -- Scale theta value to 0-255
    theta = theta*(256/(2*math.pi))
    
    -- delete colors with I and Q channel = 128
    res:at(r,c).y = theta
    res:at(r,c).yiq[1] = 128
    res:at(r,c).yiq[2] = 128
  end
  
  return il.YIQ2RGB( res )
end
----------------------------------------------

--[[
The Kirsch edge magnitude operator in LuaIP takes the maximum of the 8 template
operators, divides by 3, and then clips the result at 255. This combination of scaling and
clipping produces an edge image that generally looks better (closer to gradient magnitude
operator results) than just clipping (too bright) or just scaling (too dark).
--]]

-- Kirsch edge magnitude: max of 8 rotations --
local function kirschEdgeMag( img )
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local mask = {
    {-3, -3, -3, -3, 0, -3, 5, 5, 5 }, -- k1
    {-3, -3, -3, 5, 0, -3, 5, 5, -3 }, -- k2
    {5, -3, -3, 5, 0, -3, 5, -3, -3 }, -- k3
    {5, 5, -3, 5, 0, -3, -3, -3, -3 }, -- k4
    {5, 5, 5, -3, 0, -3, -3, -3, -3 }, -- k5
    {-3, 5, 5, -3, 0, 5, -3, -3, -3 }, -- k6
    {-3, -3, 5, -3, 0, 5, -3 ,-3, 5 }, -- k7
    {-3, -3, -3, -3, 0, 5, -3, 5, 5 }  -- k8
  }
  
  for r,c in img:pixels(1) do
    local max = 0

    local sum = {}
    for a = 1, 8 do
      sum[a] = 0
    end
    
    
    -- create array for neighborhood values
    --  (do not include center)
    local dir = {
      { img:at( r-1, c-1 ).y },   -- NW
      { img:at( r-1, c   ).y },   -- N
      { img:at( r-1, c+1 ).y },   -- NE
      { img:at( r,   c+1 ).y },   -- E
      { img:at( r+1, c+1 ).y },   -- SE
      { img:at( r+1, c   ).y },   -- S
      { img:at( r+1, c-1 ).y },   -- SW
      { img:at( r,   c-1 ).y }    -- W
    }
    
    -- The 8 rotations of 
    sum[1] = ( 5*(dir[1][1] + dir[2][1] + dir[3][1]) - 3*(dir[4][1] + dir[5][1] + dir[6][1] + dir[7][1] + dir[8][1]) )
    sum[2] = ( 5*(dir[2][1] + dir[3][1] + dir[4][1]) - 3*(dir[1][1] + dir[5][1] + dir[6][1] + dir[7][1] + dir[8][1]) )
    sum[3] = ( 5*(dir[3][1] + dir[4][1] + dir[5][1]) - 3*(dir[1][1] + dir[2][1] + dir[6][1] + dir[7][1] + dir[8][1]) )
    sum[4] = ( 5*(dir[4][1] + dir[5][1] + dir[6][1]) - 3*(dir[1][1] + dir[2][1] + dir[3][1] + dir[7][1] + dir[8][1]) )
    sum[5] = ( 5*(dir[5][1] + dir[6][1] + dir[7][1]) - 3*(dir[1][1] + dir[2][1] + dir[3][1] + dir[4][1] + dir[8][1]) )
    sum[6] = ( 5*(dir[6][1] + dir[7][1] + dir[8][1]) - 3*(dir[1][1] + dir[2][1] + dir[3][1] + dir[4][1] + dir[5][1]) )
    sum[7] = ( 5*(dir[7][1] + dir[8][1] + dir[1][1]) - 3*(dir[2][1] + dir[3][1] + dir[4][1] + dir[5][1] + dir[6][1]) )
    sum[8] = ( 5*(dir[8][1] + dir[1][1] + dir[2][1]) - 3*(dir[3][1] + dir[4][1] + dir[5][1] + dir[6][1] + dir[7][1]) )
    
    -- find which rotation had the largest sum
    for i = 1, 8 do
      if max < sum[i] then
        max = sum[i]
      end
    end
    
    -- Divide max by 3
    max = math.floor( (max / 3) + 0.5 )
    
    -- Clip results by 255
    if max > 255 then
      max = 255
    end
    
    -- Set center pixel to result and eliminate color
    res:at(r,c).yiq[0] = max
    res:at(r,c).yiq[1] = 128
    res:at(r,c).yiq[2] = 128
  end
  
  return il.YIQ2RGB( res )
end
------------------------------------

-- Kirsch edge direction (+/- 22.5deg, using the same masks) --
local function kirschEdgeDir( img )
  img = il.RGB2YIQ( img )
  local res = img:clone()
  local mask = {
    {-3, -3, -3, -3, 0, -3, 5, 5, 5 }, -- k1
    {-3, -3, -3, 5, 0, -3, 5, 5, -3 }, -- k2
    {5, -3, -3, 5, 0, -3, 5, -3, -3 }, -- k3
    {5, 5, -3, 5, 0, -3, -3, -3, -3 }, -- k4
    {5, 5, 5, -3, 0, -3, -3, -3, -3 }, -- k5
    {-3, 5, 5, -3, 0, 5, -3, -3, -3 }, -- k6
    {-3, -3, 5, -3, 0, 5, -3 ,-3, 5 }, -- k7
    {-3, -3, -3, -3, 0, 5, -3, 5, 5 }  -- k8
  }
  
  for r,c in img:pixels(1) do
    local max = 0

    local sum = {}
    for a = 1, 8 do
      sum[a] = 0
    end
    
    
    -- create array for neighborhood values
    --  (do not include center)
    local dir = {
      { img:at( r-1, c-1 ).y },   -- NW
      { img:at( r-1, c   ).y },   -- N
      { img:at( r-1, c+1 ).y },   -- NE
      { img:at( r,   c+1 ).y },   -- E
      { img:at( r+1, c+1 ).y },   -- SE
      { img:at( r+1, c   ).y },   -- S
      { img:at( r+1, c-1 ).y },   -- SW
      { img:at( r,   c-1 ).y }    -- W
    }
    
    -- The 8 rotations of 
    sum[1] = ( 5*(dir[1][1] + dir[2][1] + dir[3][1]) - 3*(dir[4][1] + dir[5][1] + dir[6][1] + dir[7][1] + dir[8][1]) )
    sum[2] = ( 5*(dir[2][1] + dir[3][1] + dir[4][1]) - 3*(dir[1][1] + dir[5][1] + dir[6][1] + dir[7][1] + dir[8][1]) )
    sum[3] = ( 5*(dir[3][1] + dir[4][1] + dir[5][1]) - 3*(dir[1][1] + dir[2][1] + dir[6][1] + dir[7][1] + dir[8][1]) )
    sum[4] = ( 5*(dir[4][1] + dir[5][1] + dir[6][1]) - 3*(dir[1][1] + dir[2][1] + dir[3][1] + dir[7][1] + dir[8][1]) )
    sum[5] = ( 5*(dir[5][1] + dir[6][1] + dir[7][1]) - 3*(dir[1][1] + dir[2][1] + dir[3][1] + dir[4][1] + dir[8][1]) )
    sum[6] = ( 5*(dir[6][1] + dir[7][1] + dir[8][1]) - 3*(dir[1][1] + dir[2][1] + dir[3][1] + dir[4][1] + dir[5][1]) )
    sum[7] = ( 5*(dir[7][1] + dir[8][1] + dir[1][1]) - 3*(dir[2][1] + dir[3][1] + dir[4][1] + dir[5][1] + dir[6][1]) )
    sum[8] = ( 5*(dir[8][1] + dir[1][1] + dir[2][1]) - 3*(dir[3][1] + dir[4][1] + dir[5][1] + dir[6][1] + dir[7][1]) )
    
    -- find which rotation had the largest sum
    for i = 1, 8 do
      if max < sum[i] then
        max = sum[i]
      end
    end

    -- Divide max by 3
    max = math.floor( (max / 3) + 0.5 )
    
    -- Clip results by 255
    if max > 255 then
      max = 255
    end
    
    local theta = 0
    -- Scale theta value to 0-255
    theta = (22.5*max)*(256/(2*math.pi))
  
    -- delete colors with I and Q channel = 128
    res:at(r,c).y = theta
    res:at(r,c).yiq[1] = 128
    res:at(r,c).yiq[2] = 128
  end
  
  return il.YIQ2RGB( res )
  
end
------------------------------------


-- Laplacian edge operator (offset by 128 and clip)
local function laplacian( img )
  img = il.RGB2YIQ( img )
  local res = img:clone()
  
  -- subtract positive laplacian operator from image
  for r,c in img:pixels(1) do
    local sum = 0
    
  
    sum = sum + img:at(r-1, c  ).y  * 1 
    sum = sum + img:at(r, c-1).y    * 1 
    sum = sum + img:at(r, c  ).y    * -4 
    sum = sum + img:at(r, c+1).y    * 1 
    sum = sum + img:at(r+1, c  ).y  * 1 
  
    
    -- offset by 128
    sum = math.floor(((sum+128) / (4)))
    
    if sum > 255 then
      sum = 255
    end
    
    res:at(r,c).y = sum
  end
  return il.YIQ2RGB( res )
end


-- embossing --
local function emboss( img )
  img = il.RGB2YIQ( img )
  local res = img:clone()
  
  for r,c in img:pixels(1) do
    local sum = 0
    sum = sum + img:at(r-1, c-1).y * -1
    sum = sum + img:at(r-1, c  ).y * -1
    sum = sum + img:at(r  , c-1).y * -1
    
    sum = sum + img:at(r  , c+1).y *  1
    sum = sum + img:at(r+1, c  ).y *  1
    sum = sum + img:at(r+1, c+1).y *  1
    
    if sum < 0 then
      sum = 0
    elseif sum > 255 then
      sum = 255
    end
    
    sum = math.abs(sum)
    
    res:at(r,c).y = sum
    --res:at(r,c).yiq[1] = 128
    --res:at(r,c).yiq[2] = 128
  end
  
  return il.YIQ2RGB( res )
end



----------
-- menus--
----------

imageMenu("Point process", 
  {
    {"Grayscale\tCtrl-G", il.grayscale, hotkey = "C-G"}
  }
)

imageMenu("Histogram Processes",
  {
    {"Display Histogram", il.showHistogram, {{name = "color model", type = "string", displaytype = "combo", choices = {"yiq", "ihs", "yuv"},default = "yiq"}}},
    {"Hist. Eq. Clipped", il.equalize, {{name = "color model", type = "string", displaytype = "combo", choices = {"yiq", "ihs", "yuv"},default = "yiq"}}},
    {"Contrast Stretch", il.stretch, {{name = "color model", type = "string", displaytype = "combo", choices = {"yiq", "ihs", "yuv"},default = "yiq"}}}
  }
)

imageMenu("Neighborhood Ops", 
  {
    {"Smooth", smooth3x3},
    {"Sharpen", sharpen3x3},
    {"Mean", mean,{{name="  n", type="number",displaytype="textbox", default = "3"}}},
    {"Median", median, {{name="  n", type="number",displaytype="textbox", default = "3"}}},
    {"Min", min,{{name="  n", type="number",displaytype="textbox", default = "3"}}},
    {"Max", max, {{name="  n", type="number",displaytype="textbox", default = "3"}}},
    {"Range", range, {{name="  n", type="number",displaytype="textbox", default = "3"}}},
    {"Median (+Shaped)", plusShape},
    {"Out-of-Range Noise", outOfRngNoiseClean,{{name = "threshold", type = "number", displaytype = "slider", default = 20, min = 0, max = 255}}},
    {"Emboss", emboss}
  }
)

imageMenu("Edge Detection",
  {
   {"Sobel Magnitude", sobelEdgeMag},
   {"Sobel Direction", sobelEdgeDir},
   {"Standard Dev.", stddev, {{name="  n", type="number",displaytype="textbox", default = "3"}}},
   {"Kirsch Mag.", kirschEdgeMag},
   {"Kirsch Dir.", kirschEdgeDir},
   {"Laplacian", laplacian}
  }
)

imageMenu("Weiss Test",
  {
    {"Sobel Mag.", il.sobelMag},
    {"Sobel Dir.", il.sobel},
    {"Std Dev", il.stdDev,
      {{name = "width", type = "number", displaytype = "spin", default = 3, min = 0, max = 65}}},
    {"Kirsch", il.kirsch}
  }
)

imageMenu("Misc",
  {
    {"Binary Threshold", bThreshold, {{name = "threshold", type = "number", displaytype = "slider", default = 128, min = 0, max = 255}}},
     {"Impulse Noise", il.impulseNoise, {{name = "probability", type = "number", displaytype = "slider", default = 64, min = 0, max = 1000}}},
     {"Gaussian Noise", il.gaussianNoise, {{name = "sigma", type = "number", displaytype="textbox", default = "16.0"}}}
  }
)

start()