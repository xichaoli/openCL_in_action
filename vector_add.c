//
// Created by lixc on 2023/12/13.
//
#include <stdio.h>
#include <CL/cl.h>
#define ARRAY_SIZE 1024

const char* programSource =
        "__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* c) {\n"
        "    int i = get_global_id(0);\n"
        "    c[i] = a[i] + b[i];\n"
        "}\n";

int main() {
    // 创建并初始化输入向量
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i;
        b[i] = i;
    }

    // 获取平台和设备信息
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, &device, NULL);

    // 创建上下文和命令队列
    const cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    const cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    // 创建内存对象
    const cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ARRAY_SIZE, NULL, NULL);
    const cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ARRAY_SIZE, NULL, NULL);
    const cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * ARRAY_SIZE, NULL, NULL);

    // 将数据从主机内存复制到设备内存
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, b, 0, NULL, NULL);

    // 创建程序对象并编译内核
    const cl_program program = clCreateProgramWithSource(context, 1, &programSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // 创建内核对象
    const cl_kernel kernel = clCreateKernel(program, "vectorAdd", NULL);

    // 设置内核参数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // 启动内核执行
    const size_t globalSize = ARRAY_SIZE;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // 从设备内存读取结果到主机内存
    float c[ARRAY_SIZE];
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, c, 0, NULL, NULL);

    // 打印结果
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }

    // 释放资源
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
