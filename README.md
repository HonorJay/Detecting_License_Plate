original source from https://github.com/chequanghuy/Character-Time-series-Matching

# git clone 
```python
!git clone https://github.com/HonorJay/Detecting_License_Plate.git
```

# change directory
```python
!cd /Detecting_License_Plate/Vietnamese/
```

# to Detect license plate and characters from video.
```python
!python DETECTION.py --lp_weights object.pt --ch_weights char.pt --source test_video.mp4 --device cuda:0
```

# result.json
<div align=center>
<img src='result.png' width='300'>
</div>
