cmd = "ffmpeg -y -r 5 -i pcp_%03d.png -c:v libx264 -r 20 -vf scale=2052:1004 {}.mp4"
os.system(cmd.format('../../Results/tmp/pcp'))
