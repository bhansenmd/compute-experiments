import time
import math
import array

from vulkan import *
from PIL import Image
import numpy as np


WIDTH = 3200
HEIGHT = 2400
WORKGROUP_SIZE = 32

ENABLE_VALIDATION_LAYERS = True


def main():

    instance = None
    debugReportCallback = None
    physicalDevice = None
    device = None

    pipeline = None
    pipelineLayout = None
    computeShaderModule = None

    commandPool = None
    commandBuffer = None

    descriptorPool = None
    descriptorSet = None
    descriptorSetLayout = None

    buffer = None
    bufferMemory = None

    bufferSize = 0
    enabledLayers = []

    queue = None

    queueFamilyIndex = -1

    pixel = array.array('f', [0, 0, 0, 0])

    saveImageTime = 0
    cpuDataConverTime = 0



    # cleanup

    if ENABLE_VALIDATION_LAYERS:
        # destroy callback.
        func = vkGetInstanceProcAddr(instance, 'vkDestroyDebugReportCallbackEXT')
        if func == ffi.NULL:
            raise Exception("Could not load vkDestroyDebugReportCallbackEXT")
        if debugReportCallback:
            func(instance, debugReportCallback, None)

    if bufferMemory:
        vkFreeMemory(device, bufferMemory, None)
    if buffer:
        vkDestroyBuffer(device, buffer, None)
    if computeShaderModule:
        vkDestroyShaderModule(device, computeShaderModule, None)
    if descriptorPool:
        vkDestroyDescriptorPool(device, descriptorPool, None)
    if descriptorSetLayout:
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, None)
    if pipelineLayout:
        vkDestroyPipelineLayout(device, pipelineLayout, None)
    if pipeline:
        vkDestroyPipeline(device, pipeline, None)
    if commandPool:
        vkDestroyCommandPool(device, commandPool, None)
    if device:
        vkDestroyDevice(device, None)
    if instance:
        vkDestroyInstance(instance, None)


if __name__ == '__main__':
    main()
