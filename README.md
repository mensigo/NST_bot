## Neural Style Transfer

Platform: Telegram<br> @LonAloneBot
Framework: aiogram<br>
Algorithm: 1) standard, vgg-19 based, cut version
              https://arxiv.org/abs/1508.06576
           2) cycle-gan: junyanz/pytorch-CycleGAN-and-pix2pix
              https://arxiv.org/pdf/1703.10593.pdf
Commands:

/run - get instructions to do style transfer (your style):
  a) enter /run
  b) send photo to transfer style on \[content photo\]
  c) send photo to get style from \[style photo\]
  d) wait ~90 sec
  e) receive result image
  
/ukiyoe - same but fixed (ukiyoe) style:
  a) enter /ukiyoe
  b) send photo to transfer style on \[content photo\]
  c) wait ~7 sec

/stopit - return to the starting point (reset state)
/help - about me & procedure

Details:
- images are resized to 256 (larger side) if necessary
- gan resizes to a 256 square
- gan inference time includes downloading weights (40+ mb)
