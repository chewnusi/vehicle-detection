docker run --rm -it --network=host bluenviron/mediamtx:latest

ffmpeg -re -stream_loop -1 -i /home/chewnusi/diploma/videos/apc.mp4 -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -c:a aac -b:a 128k -f rtsp rtsp://127.0.0.1:8554/live/vehicles_stream

ffplay rtsp://127.0.0.1:8554/live/vehicles_stream