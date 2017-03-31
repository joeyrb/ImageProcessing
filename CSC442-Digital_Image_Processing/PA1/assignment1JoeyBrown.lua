--[[
Author: Joey Brown
Class: CSC 442, Digitial Image Processing
Asgmnt: PA #1

Sources:
  * Menu items were modeled after the lip.lua examples written by Dr. Weiss and Alex Iverson
  * Posterize help: http://www.axiomx.com/posterize.htm
  * Color LUT: https://en.wikipedia.org/wiki/Colour_look-up_table (used in dPseudocolor)
  * Histogram Eq: https://www.youtube.com/watch?v=eNBZI-qYhpg
--]]

require "ip"
local viz = require "visual"
local il = require "il"

local imgs = {...}
for i, fname in ipairs(imgs) do loadImage(fname) end



---------------
-- functions --
---------------

---- POINT PROCESSES --
-- grayscale--
-- uses human perception of color to change rgb values to gray w/ varying intensities
local function imgGrayscale( img ) 
  
  local gray
  for r,c in img:pixels() do
    gray = img:at(r,c).r*0.3+img:at(r,c).g*0.59+img:at(r,c).b*0.11
    img:at(r,c).r = gray
    img:at(r,c).g = gray
    img:at(r,c).b = gray
  end

  return img
end
-----------------------------------

-- negate --
-- subtract grayscale intensity of image from 255 then convert back to color
local function imgNegate ( img )
  local pix
  for r,c in img:pixels() do 
    pix = img:at(r,c)
    pix.r = 255 - pix.r
    pix.g = 255 - pix.g
    pix.b = 255 - pix.b
  end
  
  return img -- this works because pix acts like a pointer
end
  -----------------------------------
  
-- posterize --
local function posterize ( img, nlvls )
  local nAreas = math.floor( 256 / nlvls )
  local nVals = math.floor( 255 / (nlvls-1) )
  
  for r,c in img:pixels() do
    for ch = 0, 2 do
      local clr = math.floor( img:at(r,c).rgb[ch] / nAreas )
      if( math.floor( nVals * clr) <= 255 ) then
        img:at(r,c).rgb[ch] = math.floor( nVals * clr )
      else
        img:at(r,c).rgb[ch] = 255
      end
    end
  end
  return img
end
-----------------------------------

-- log transformation --
--[[
    The general form of the log transformation is s= c*log(1 + r) where c is a constant, 
    and it is assumed that r >= 0. [...] We use a transformation of this type to exapnd 
    the values of dark pixels in an image while compressing the higher-level values. The 
    opposite is true of the inverse log transformation.(Digital Image Processing, pg. 131).
    
    The equation, ( log(i+1) / log(256) ) * 255 was used instead where i = intensity input
    
--]]
local function logTransform( img )
  -- convert img RGB -> YIQ
  img = il.RGB2YIQ( img )
  
  -- perform log transform
  for r,c in img:pixels() do
    img:at(r,c).y = math.log(img:at(r,c).y+1) / math.log(256) * 255
  end
  
  -- convert img YIQ -> RGB and return img
  return il.YIQ2RGB( img )
end
-----------------------------------

-- brightness adjust --
local function brightness( img, input, model )
  img = il.RGB2YIQ( img )
  local pix
  
  for r,c in img:pixels() do
    pix = img:at(r,c).y
    pix = pix + input
    
    -- check upper bounds
    if (pix + input > 255) then 
      pix = 255
    end
    
    -- check lower bounds
    if (pix + input < 0) then
      pix = 0
    end
    
    img:at(r,c).y = pix
  end

  return il.YIQ2RGB( img )
end
-----------------------------------

-- contrast adjust --
-- uses an equation to change the histogram ramp then clips min and max
local function adjContrast( img, imin, imax)
  --img = il.RGB2IHS( img )
  img = il.RGB2YIQ( img )
  
  
  for r,c in img:pixels() do
    
    local i = (255/(imax-imin)) * (img:at(r,c).y - imin)
    
    if i > 255 then       -- clip at upper bound
      img:at(r,c).y = 255
    elseif i < 0 then
      img:at(r,c).y = 0   -- clip at lower bound
    else
      img:at(r,c).y = i   -- map pixels
    end
  end
  
  --return il.IHS2RGB( img )
  return il.YIQ2RGB( img )
end
-----------------------------------

-- gamma power transformation --
local function gammaPwrTrns( img, ingamma )
  img = il.RGB2YIQ( img )
  
  for r,c in img:pixels() do 
    local pix = img:at(r,c).y             -- get intensity
    pix = pix / 255                       -- set intensity to percentage
    pix = math.floor(255 * (pix)^ingamma) -- normalize pix to convert back into intensity
    img:at(r,c).y = pix                   -- set intensity equal to pix
  end
  
  return il.YIQ2RGB( img )                -- convert intensities to rgb for display
end
-----------------------------------

-- discrete 8-level pseudocolor -- 
--[[Duplicate image (dup), convert img to IHS,
create ranges for intensity mapping (var[])(8-bit),
convert dup into RGB, measure the intensity of
img's pixels(pix), if intensity is within 0-32 then
index is set to 1, if intensity is >32 then index is 
incremented until pix <= val[index], use the value of
val[index] with the rgbLUT to map the rgb values of 
dup, return the manipulated image (dup). --]]
local function dPseudocolor( img )
  img = il.RGB2IHS( img ) -- convert to IHS to find intensity
  local dup = img:clone()
  dup = il.IHS2RGB( dup ) -- convert clone to RGB to map color

  -- 8-bit LUT --
  local rgbLUT = {
        [ 0 ]  = { 0,   0,    0   },
        [ 1 ]  = { 0,   0,    0   },
        [ 2 ]  = { 0,   0,    255 },
        [ 3 ]  = { 0,   255,  0   },
        [ 4 ] = { 0,   255,  255 },
        [ 5 ] = { 255, 0,    0   },
        [ 6 ] = { 255, 0,    255 },
        [ 7 ] = { 255, 255,  0   },
        [ 8 ] = { 255, 255,  255 }
    }
  
  -- TODO:
  -- to avoid nested loop, math.floor(intensity) / 32
  --  then use LUT to map intensities to color values
  -- [math.floor() creates steps on the histogram]
  
  -- find level of intensity --
  for r,c in img:pixels() do
    local pix = math.floor(img:at(r,c).ihs[0] / 32) + 1
    img:at(r,c).rgb = rgbLUT[pix]
  end
  
  --return dup
  return img
end
-----------------------------------

-- continuous pseudocolor --
--[[ uses sin and cos to map a range of color values to grayscale intensities
--]]
local function cPseudocolor ( img ) 
 -- initialize LUT --
 local LUT = {}
 for i=0, 255 do
    -- convert from into to radian (phase conversion)
    local phase = (i / 255) * (2* math.pi)
    
    -- f(i) can use any equation for each r,g,b
    local r = math.cos(phase)
    local g = math.sin(phase)
    local b = i
    -- convert back into range [0,255] for output
    r = (r+1) * ( 255 / 2 )
    g = (g+1) * ( 255 / 2 )
    
    LUT[i] = {r, g, b}
    
 end
  
  -- duplicate img to measure intensity and convert to rgb --
  local dup = img:clone()
  img = il.RGB2IHS( img )
  
  -- loop through all pixels in the image -- 
  for r,c in img:pixels() do
    local pix = img:at(r,c).ihs[0] -- get intensity
    dup:at(r,c).rgb = LUT[pix] -- assign intensity to LUT
  end
    
   return dup
end
-----------------------------------

-- bitplane slicing --
--[[ uses the bitwise AND function to preserve intesities,
 preserved intensities are made white and everything else is
 set to black.
--]]
local function bpSlice( img, plane )
  local lvlLUT = {2, 4, 8, 16, 32, 64, 128, 256}
  img = il.RGB2IHS( img ) -- convert to IHS to measure intensity
  
  -- bitwise AND the pixels with the bit plane input --
  for r,c in img:pixels() do
    local pix = img:at(r,c).ihs[0] -- used pix to reduce typing
    if bit.band(pix, lvlLUT[plane]) == 0 or pix == 0 then
      img:at(r,c).ihs[0] = 0    -- the bit = plane is not preserved
    else
      img:at(r,c).ihs[0] = 255  -- the bit = plane is preserved
    end
  end  
  img = il.IHS2RGB( img ) -- convert back to RGB for display
  img = imgGrayscale( img ) -- remove color with grayscale function
  return img
end
-----------------------------------


-- HISTOGRAM --
-- histogram equalization --
--[[ measures intensity (8-bit) from [0,255], plots a histogram,
  creates CDF based on histogram, maps new histogram to the image
--]]
local function histogramEq( img )
  img = il.RGB2IHS( img )
  local totalPix = img.height*img.width
  -- initialize histogram --
  local h = {}
  for i = 0, 255 do
    h[i] = 0
  end

  -- find frequency histogram --
  local fsum = 0.0
  for r,c in img:pixels() do
    local indx = img:at(r,c).ihs[0] -- get intensity val to index
    local x = h[indx]
    h[indx] = x + 1                 -- increment probability to histogram
  end

  -- create CDF LUT --
  local LUT, prob, sum = {}, {}, 0.0
  for j = 0, 255 do
    prob[j] = h[j]/totalPix
    sum = sum + prob[j]
    LUT[j] = sum
  end

  
  -- map intensities --
  for r,c in img:pixels() do
    local pix = img:at(r,c).ihs[0]
    local n = LUT[pix]
    img:at(r,c).ihs[0] = math.floor(n*255)
  end
  
  return il.IHS2RGB( img )
end
-----------------------------------

-- histogram eq. w/ clipping -- 
-- get clipping threshold from user as % --
-- uses the same method as histogramEq() --
-- except it uses a percentage to clip   --
-- the frequency of an intensity by %    --
local function histeqClipped ( img, thrshPct )
  local totalPix = img.height*img.width
  thrshPct = thrshPct/100 -- convert decimal into percent
  local clip = totalPix*thrshPct -- max frequency
  img = il.RGB2IHS( img )
  
  -- initialize histogram --
  local h = {}
  for i = 0, 256 do
    h[i] = 0
  end

  -- find frequency histogram --
  local fsum = 0
  for r,c in img:pixels() do
    local indx = img:at(r,c).ihs[0] -- get intensity val for index
    local x = h[indx]
    h[indx] = x + 1
    if h[indx] > clip then
      h[indx] = clip                 -- increment probability to histogram
      fsum = fsum + 1
    end
  end
  print(255/fsum)

  -- create CDF LUT --
  local LUT, sum = {}, 0.0
  LUT[0] = math.floor( ((255/fsum)*h[0])+0.5 )
  for j = 1, 255 do
    sum = sum + (h[j]/totalPix)
    LUT[j] = math.floor( (LUT[j-1]+(255/fsum)*h[j])+0.5)
  end

  -- map intensities --
  for r,c in img:pixels() do
    local pix = img:at(r,c).ihs[0]
    img:at(r,c).ihs[0] = LUT[pix]
  end
  
  return il.IHS2RGB( img )
end
-----------------------------------

-- automated contrast stretch --
local function autoContrast( img )
  img = il.RGB2YIQ( img )
  local min, max = 255, 0
  
  -- get min/max values --
  for r,c in img:pixels() do
    -- find min --
    if img:at(r,c).y < min then
      min = img:at(r,c).y
    end
    -- find max --
    if img:at(r,c).y > max then
      max = img:at(r,c).y
    end
  end
  
  -- apply contrast stretch to img --
  return adjContrast( il.YIQ2RGB( img ), min, max )
end
-----------------------------------

-- modified contrast stretch --
local function modContrast( img, minPct, maxPct )
  -- convert input to percentage
  local min, max = math.floor(255*(minPct/100)), math.floor(255*(maxPct/100))
  return adjContrast( img, min, max )
end
-----------------------------------


-- MISC --
-- binary threshold --
local function bThreshold ( img, input )
  
  img = il.RGB2YIQ( img )
  local res = img:clone()
  --local pix 
  for r,c in res:pixels() do
    --pix = img:at(r,c).y
    if ( img:at(r,c).y >= input) then
      res:at(r,c).r = 255
      res:at(r,c).g = 255
      res:at(r,c).b = 255
      --pix.r = 255
      --pix.g = 255
      --pix.b = 255
      --res:at(r,c).b = 0
    else
      res:at(r,c).r = 0
      res:at(r,c).g = 0
      res:at(r,c).b = 0
      --res:at(r,c).b = 1
      --pix.r = 0
      --pix.g = 0
      --pix.b = 0
    end
    
  end
  --return il.YIQ2RGB( img )
  img = res
  return img
end
-----------------------------------

-- flip image horizontally --
local function imgFlipH( img )
  local flip, h, w = img:clone(), img.height-1, img.width-1
  
  for r,c in img:pixels() do
    flip:at(h-r,w-c).rgb = img:at(r,c).rgb
  end
  
  return flip
end
-----------------------------------

-- flip image vertically --
local function imgFlipV( img )
  local flip, w  = img:clone(), img.width-1
  
  for r,c in img:pixels() do
    flip:at(r,w-c).rgb = img:at(r,c).rgb
  end
  
  return flip
end

-----------------------------------

-- random discrete pseudocolor --
-- ( point process of choice ) --
local function randPseudoClr( img )
  img = il.RGB2IHS( img ) -- convert to IHS to find intensity
  local dup = img:clone()
  dup = il.IHS2RGB( dup ) -- convert clone to RGB to map color

  -- set up ref table for 8-level graph --
  local val = {}
  for i = 1, 8 do 
    val[i] = i * 32
  end
  val[8] = 255  -- change 256 to 255
  
  -- create 8-bit LUT with random values set for each rgb channel --
  math.randomseed( os.time() )
  local LUT = {
      [ 32 ]  = { 0, 0, 0 },
      [ 64 ]  = { 0, 0, math.random(0,255) },
      [ 96 ]  = { 0, math.random(0,255), 0   },
      [ 128 ] = { 0, math.random(0,255), math.random(0,255) },
      [ 160 ] = { math.random(0,255), 0, 0   },
      [ 192 ] = { math.random(0,255), 0, math.random(0,255) },
      [ 224 ] = { math.random(0,255), math.random(0,255), 0   },
      [ 255 ] = { math.random(0,255), math.random(0,255), math.random(0,255) }
  }
  
  -- find level of intensity --
  for r,c in img:pixels() do
    local pix = img:at(r,c).ihs[0]
    
    -- find the range of intensity at (r,c)
    local index = 1
    if( pix > 32 ) then 
      while( pix > val[index] and index < 8 ) do
        index = index + 1 -- increment index to move to next "step" in the graph
      end
    else
      index = 1
    end
    
    -- use rgb LUT to replace grayscale w/ color --
    dup:at(r,c).rgb = LUT[val[index]]
    
  end
  
  return dup
end
-----------------------------------


----------
-- menus--
----------

imageMenu("Point process", 
  {
    {"Grayscale\tCtrl-G", imgGrayscale, hotkey = "C-G"},
    {"Negate\tCtrl-N", imgNegate, hotkey = "C-N"},
    {"Posterize", posterize, {{name = "levels", type = "number", displaytype = "spin", default = 8, min = 2, max = 255}}},
    {"Logarithmic Transform", logTransform},
    {"Brightness", brightness, {{name = "value", type = "number", displaytype = "slider", default = 0, min = -255, max = 255}}},
    {"Contrast Stretch", adjContrast, {{name = "min", type = "number", displaytype = "spin", default = 32, min = 2, max = 255},
                                       {name = "max", type = "number", displaytype = "spin", default = 128, min = 2, max = 255}}},
    {"Gamma", gammaPwrTrns, {{name="gamma", type = "string", default = "1.0"}}},
    {"Discrete Pseudocolor", dPseudocolor},
    {"Continuous Pseudocolor", cPseudocolor},
    {"Bitplane slicing", bpSlice, {{name = "plane ", type = "number", displaytype = "spin", default = 7, min = 0, max = 7}}}
  }
)

imageMenu("Histogram Processes",
  {
    {"Display Histogram", il.showHistogram, {{name = "color model", type = "string", default = "ihs"}}},
    {"Histogram Equalizatin", histogramEq},
    {"Hist. Eq. Clipped", histeqClipped, {{name = "clip %", type = "number", displaytype = "textbox", default = "1.0"}}},
    {"Automatic Contrast Stretch", autoContrast},
    {"Modified Contrast Stretch", modContrast, {{name = "dark pixels (%) ", type = "number", displaytype = "spin", default = 1, min = 0, max = 100},
                                                {name = "light pixels (%) ", type = "number", displaytype = "spin", default = 99, min = 0, max = 100}}}
  }
)

imageMenu("Misc",
  {
    {"Binary Threshold", bThreshold, {{name = "threshold", type = "number", displaytype = "slider", default = 128, min = 0, max = 255}}},
    {"Flip image Horiz.", imgFlipH},
    {"Flip image Vert.", imgFlipV},
    {"Random Discrete Pseudocolor", randPseudoClr}
  }
)

start()