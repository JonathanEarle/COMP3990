import math
import os
import random
from random import randint
import sys
import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor

FONT_DIR="./fonts"
FONT_HEIGHT=32
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = DIGITS+LETTERS+" "
WIDTH=160
HEIGHT=80

#Turn plate number into an image
def MakeImg(shade,background,path,Outheight):
	Img=[]
	fontSize=Outheight*4
	font=ImageFont.truetype(path,fontSize)

	height=max(font.getsize(c)[1] for c in CHARS)

	for c in CHARS:
		width=font.getsize(c)[0]
		img=Image.new("RGBA",(width,height),(background,background,background))

		draw=ImageDraw.Draw(img)
		draw.text((0,0),c,(shade,shade,shade),font=font)
		scale=float(Outheight)/height
		img=img.resize((int(width*scale),Outheight),Image.ANTIALIAS)
		Img.append((c,numpy.array(img)[:,:,0].astype(numpy.float32)))
	return Img

#Loads the fonts
def loadFonts(path):
	fontImg={}
	fonts=[]
	col=randint(0,1)
	shade=col*255
	background=(1-col)*255
	for file in os.listdir(path):
		if file.endswith('.ttf'):
			fonts.append(file)
	for font in fonts:
		fontImg[font]=dict(MakeImg(shade,background,os.path.join(path,font),FONT_HEIGHT))

	return fonts,fontImg,background

def mergeImages(front,back):	
	foreground=Image.fromarray(front)
	background=Image.fromarray(back,'RGB')
	fw,fh=foreground.size
	bw,bh=background.size	

	#Resize front randomly
	fw=int(fw/random.uniform(1,2))
	fh=int(fh/random.uniform(1,2))
	foreground=foreground.resize((fw,fh))

	#Tilt front randomly
	tempBackground=Image.fromarray(generateBackground(2*HEIGHT,2*WIDTH),'RGB')
	offset=(((2*WIDTH-fw)/2),((2*HEIGHT-fh)/2))
	tempBackground.paste(foreground,offset)
	tempBackground=tempBackground.rotate(randint(-10,10))
	foreground=tempBackground	
	fw,fh=foreground.size

	#Skew front randomly
	foreground.transform((fw,fh),method=Image.AFFINE,data=(1,0.5,randint(0,WIDTH),0,1,0))

	#Place front randomly
	offset=(((bw-fw)/2)+randint(-5,5),((bh-fh)/2)+randint(-5,5))
	background.paste(foreground,offset)

	return numpy.array(background)

def generateBackground(h,w):
	return numpy.random.rand(h,w,3) * 255

#Generates a random plate number
def generateCode():
	cat=randint(0,2)
	if cat==0:
		return "{}{}{} {}{}{}{}".format(
			random.choice(LETTERS),
			random.choice(LETTERS),
			random.choice(LETTERS),
			random.choice(DIGITS),
			random.choice(DIGITS),
			random.choice(DIGITS),
			random.choice(DIGITS))
	elif cat==1:
		return " {}{}{}\n{}{}{}{}".format(
			random.choice(LETTERS),
			random.choice(LETTERS),
			random.choice(LETTERS),
			random.choice(DIGITS),
			random.choice(DIGITS),
			random.choice(DIGITS),
			random.choice(DIGITS))
	
	return "{}{} {}{}{}{}".format(
		random.choice(LETTERS),
		random.choice(LETTERS),
		random.choice(DIGITS),
		random.choice(DIGITS),
		random.choice(DIGITS),
		random.choice(DIGITS))

def generatePlate(font_height, char_ims, code,background):
	h_padding = random.uniform(0.2, 0.4) * font_height
	v_padding = random.uniform(0.1, 0.3) * font_height
	spacing = font_height * random.uniform(-0.002, 0.005)
	radius = 1 + int(font_height * 0.1 * random.random())
	
	multi=False
	if '\n' in code:
		multi=True
		code=code.split('\n')
		code2=code[1]
		code=code[0]

	text_width = sum(char_ims[c].shape[1] for c in code)	
	text_width += (len(code) - 1) * spacing

	if multi==True:
		text_width2 = sum(char_ims[c].shape[1] for c in code2)	
		text_width2 += (len(code2) - 1) * spacing
		out_shape = (int(font_height+font_height + v_padding * 2),int(text_width2 + h_padding * 2))
	else:
		out_shape = (int(font_height + v_padding * 2),int(text_width + h_padding * 2))

	plate=numpy.ones(out_shape)*background

	x = h_padding
	y = v_padding
	for c in code:		
		char_im = char_ims[c]
		ix, iy = int(x), int(y)
		if(ix + char_im.shape[1]>plate.shape[1]):
			ix=ix-((ix + char_im.shape[1])-plate.shape[1])
		plate[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
		x += char_im.shape[1] + spacing

	x = h_padding
	y = v_padding
	if multi==True:
		for c in code2:		
			char_im = char_ims[c]
			ix, iy = int(x), int(y)
			if(ix + char_im.shape[1]>plate.shape[1]):
				ix=ix-((ix + char_im.shape[1])-plate.shape[1])
			plate[iy+int(font_height):iy+int(font_height) + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
			x += char_im.shape[1] + spacing

	return plate

def generateImage(code):
	fonts, fontChars, background=loadFonts(FONT_DIR)
	plate= generatePlate(FONT_HEIGHT, fontChars[random.choice(fonts)], code, background)
	bg=generateBackground(HEIGHT,WIDTH)
	image=mergeImages(plate,bg)
	return image	

#Generates a random tuple of a plate number and a corresponding image of that plate number
def generateData():
	code=generateCode()
	image=generateImage(code)
	return (image,code)

def main():
	for i in range(0,int(sys.argv[1])):
		image,code=generateData()
		file="test/{}.png".format(code)
		#print(code)
		cv2.imwrite(file,image)

if __name__=="__main__":
	main()
