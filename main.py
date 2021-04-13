# -*- coding: utf-8 -*-
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import subprocess
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
import numpy as np
from argparse import ArgumentParser
from inference import Network
from csv import DictWriter
from datetime import datetime
from collections import deque

VIDEO_PATH = "resources/Pedestrian_Detect_2_1_1.mp4"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")

    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    # Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_bboxes(frame,output_result,prob_threshold, width, height):
    """
    Draw bounding boxes onto the frame.
    :param frame: frame from camera/video
    :param result: list contains the data comming from inference
    :return: person count and frame
    """
    counter=0
    start = None
    end = None 
    color = (0, 255, 0)
    thickness = 1
    for box in output_result[0][0]:
        if box[2] > prob_threshold:
            start = (int(box[3] * width), int(box[4] * height))
            end = (int(box[5] * width), int(box[6] * height))
            frame = cv2.rectangle(frame, start, end, color,thickness)
            counter+=1
    return frame, counter

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    plugin = Network()

    # Set Probability threshold for detections
    if not args.prob_threshold is None:
        prob_threshold = args.prob_threshold
    else:
        prob_threshold = 0.5

    # Load the model through `infer_network`
    plugin.load_model(args.model, args.cpu_extension, args.device)
    net_input_shape = plugin.get_input_shape()

    # Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
        single_image_mode = False
    elif args.input[-4:] in [".jpg", ".bmp", ".png"]:
        single_image_mode = True
        input_stream = args.input
    else:
        single_image_mode = False
        input_stream = args.input
        assert os.path.isfile(input_stream)

    
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    if not cap.isOpened():
        print("Unable open video stream")

    data_list = []
    inference_t = 0
    process_t = 0
    prev_count = 0
    total_count = 0
    duration = 0
    request_id = 0

    while cap.isOpened():
        log_data = {}
        flag, frame = cap.read()

        if not flag:
            sys.stdout.flush()
            break

        width = int(cap.get(3))
        height = int(cap.get(4))

        # Preprocessing / Might be affected by Model choice.
        p_frame = cv2.resize(frame, (net_input_shape[3],net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Start asynchronous inference
        t0 = time.time()
        plugin.exec_net(p_frame, request_id)

        # Wait for the results
        if plugin.wait() == 0:

            # Get the results of the inference request
            result = plugin.get_output()
            t1 = time.time()
            inference_t = t1 - t0

            # Extract any desired stats from the results and publish to MQTT Server
            displayFrame, cur_count = draw_bboxes(frame, result, prob_threshold, width, height)
            inference_t_message = "Inference time: {:.3f}ms"\
                               .format(inference_t)
            cv2.putText(displayFrame, inference_t_message, (15, 15),
                       cv2.FONT_HERSHEY_COMPLEX, 0.45, (200, 10, 10), 1)

            if cur_count > prev_count:
                start_time = time.time()
                total_count = total_count + cur_count - prev_count
                client.publish("person", json.dumps({"total": total_count}))

            if cur_count < prev_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
                
            # Calculate and send relevant information on #
            # current_count, total_count and duration to the MQTT server #
            # Topic "person": keys of "count" and "total" #
            client.publish("person", json.dumps({"count": cur_count}))
            prev_count = cur_count
            
        log_data['time'] = time.strftime("%H:%M:%S", time.localtime())
        log_data['count'] = cur_count
        log_data['total_count'] = total_count
        log_data['duration'] = duration
        log_data['inference_t'] = inference_t
        log_data['process_t'] = process_t
        log_data['result'] = result
        data_list.append(log_data)

        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            write_csv(data_list)
            print('Write CSV')
            cap.release()
            cv2.destroyAllWindows()
            client.disconnect()
            break

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(displayFrame)
        sys.stdout.flush()

        # Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite("output.jpg", displayFrame)

    write_csv(data_list)
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def write_csv(data):
    with open('./log.csv', 'w') as outfile:
        writer = DictWriter(outfile, ('time', 'count',
                                      'total_count', 'duration',
                                      'inference_t', 'process_t', 'result'))
        writer.writeheader()
        writer.writerows(data)

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Connect to the MQTT server
    client = connect_mqtt()

    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()