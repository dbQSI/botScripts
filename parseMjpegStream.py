import cv2
import requests
import numpy as np
import sys
import time
from datetime import datetime

class cv2StreamParse:
    def __init__(self, URL):
        self.stream = requests.get(URL, stream=True)
        if(self.stream.status_code == 200):
            return
        else:
            raise Exception("Received unexpected status code {}".format(self.stream.status_code))

    def UtcNow(self):
        now = datetime.utcnow()
        return now.strftime("%Y-%m-%d_%H-%M-%S")

    def runStreamParseJpgs(self, outPath, chunkSize=1024):
        camBytes = bytes()
        count = 0
        for chunk in self.stream.iter_content(chunk_size=chunkSize):
            camBytes += chunk
            a = camBytes.find(b'\xff\xd8')
            b = camBytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = camBytes[a:b+2]
                camBytes = camBytes[b+2:]
                i = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(outPath + "/snapshot_" + self.UtcNow() + ".jpg", i)
                count += 1

                # cv2.imshow('i', i)
                # if cv2.waitKey(1) == 27:
                #     sys.exit(0)

if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(description='input an mjpeg stream URL and an output location and jpegs will be continuously parsed from the stream.  To stop streaming jpegs, simply press "Ctrl-C" in the terminal to halt the script.')
    argGroup = parser.add_argument_group(title='Inputs')
    argGroup.add_argument('-url', dest='urlPath', required=True, help='input URL address to mjpeg stream location.')
    argGroup.add_argument('-dir', dest='jpegDir', required=True, help='input directory path to output location for jpegs.')
    argGroup.add_argument('-chunk', required=False, default=500000, help='OPTIONAL: set chunk size for requests.iter_content parser.')
    # argGroup.add_argument('-odir', dest='outDir', required=True, help='input directory path to output location for georeferenced jpegs.')
    # argGroup.add_argument('-clahe', action='store_true', required=False, default=False, help='Flag to run OpenCV CLAHE processing on imagery as well.')
    # argGroup.add_argument('-cb', action='store_true', required=False, default=False, help='Flag to run OpenCV saturation adjustment on imagery while doing clahe processing.')
    # argGroup.add_argument('-p', dest='processes', required=False, help='number of processes allowed by user')
    args = parser.parse_args()

    if args.urlPath:
        parseStream = cv2StreamParse(args.urlPath)
        parseStream.runStreamParseJpgs(args.jpegDir)
    else:
        print("no input entered.  run with '-h' to learn about required inputs.")
