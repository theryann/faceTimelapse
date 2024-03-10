## face timelapse
### description
Run this align all pictures of your face.
I wrote this, to **not need a ton of dependencies** like other implementation do.

### HOW TO RUN
1. in the root directory create two folders:
    - pictures
    - out
2. paste your images into `pictures`
3. run `python align_photos.py` to create the stabilized pictures
4. if you added more pics to pictures you don't need to redo all picures, just use `python align_photos.py --append`
5. the aligned phots appear in 'out'.


### requirements
```bash
pip install opencv-python
pip install numpy
```


#### Note:
While this might not be the cleanest implementation, it works as expected without many imports.
I'll propably improve it sometimes.