## Neural Style Transfer

Platform: Telegram @LonAloneBot<br>
Framework: aiogram<br>
Algorithm: 1) standard, vgg-19 based, cut version
              https://arxiv.org/abs/1508.06576
           2) cycle-gan: junyanz/pytorch-CycleGAN-and-pix2pix
              https://arxiv.org/pdf/1703.10593.pdf
Commands:

/run - get instructions to do style transfer (your style):<br>
  a) enter /run<br>
  b) send photo to transfer style on \[content photo\]<br>
  c) send photo to get style from \[style photo\]<br>
  d) wait ~90 sec<br>
  e) receive result image<br>
  
/ukiyoe - same but fixed (ukiyoe) style:<br>
  a) enter /ukiyoe<br>
  b) send photo to transfer style on \[content photo\]<br>
  c) wait ~7 sec<br>

/stopit - return to the starting point (reset state)<br>
/help - about me & procedure

Details:<br>
- images are resized to 256 (larger side) if necessary
- gan resizes to a 256 square
- gan inference time includes downloading weights (40+ mb)
