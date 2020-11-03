// Bresenham's line algorithm
// Created by orz on 2020/10/20.
// 参考资料：
// https://zh.wikipedia.org/wiki/%E5%B8%83%E9%9B%B7%E6%A3%AE%E6%BC%A2%E5%A7%86%E7%9B%B4%E7%B7%9A%E6%BC%94%E7%AE%97%E6%B3%95
// http://www.lucienxian.top/2017/09/25/Bresenham-s-line-algorithm/
#include <map>
#include <iostream>
#include <eigen3/Eigen/Core>

using namespace Eigen;
using namespace std;

void draw_line_1(MatrixXf &p, int xb, int yb, int xe, int ye);

void draw_line_2(MatrixXf &p, int xb, int yb, int xe, int ye);

void draw_line_3(MatrixXf &p, int xb, int yb, int xe, int ye);

void draw_line_4(MatrixXf &p, int xb, int yb, int xe, int ye);

int main() {
    MatrixXf p(30, 30);
    p.setZero();
    int xb = 0, yb = 0,
            xe = 29, ye = 13;

    draw_line_3(p, xb, yb, xe, ye);
    cout << p << endl;
    return 0;
}

void draw_line_1(MatrixXf &p, int xb, int yb, int xe, int ye) {
    int dx = xe - xb, dy = ye - yb;
    float k = (float) dy / dx, error = 0;
    for (int i = xb, j = yb; i < xe; i++) {
        p(i, j) = 1;
        error += k;
        if (error > 0.5) {
            j++;
            error -= 1;
        }
    }
}

void draw_line_2(MatrixXf &p, int xb, int yb, int xe, int ye) {

    bool steep = abs(ye - yb) > abs(xe - xb);// 判断斜率是否大于1
    if (steep) {  // 如果斜率大于1，先计算镜像
        swap(xb, yb);
        swap(xe, ye);
    }
    if (xb > xe) {  // 如果要从高到低绘制，等价于从低到高绘制
        swap(xb, xe);
        swap(yb, ye);
    }

    int dx = xe - xb, dy = abs(ye - yb), ystep = 1;
    if (y1 < y0) ystep = -1;
    float error = 0, k = (float) dy / dx;

    for (int i = xb, j = yb; i < xe; i++) { // 不能为i != xe，因为xe可能为小数！
        if (steep) p(i, j) = 1;
        else p(j, i) = 1;
        error += k;
        if (error > 0) {
            j += ystep;
            error -= 1;
        }
    }
}

void draw_line_3(MatrixXf &p, int xb, int yb, int xe, int ye) {
    bool steep = abs(ye - yb) > abs(xe - xb);// 判断斜率是否大于1
    if (steep) {  // 如果斜率大于1，先计算镜像
        swap(xb, yb);
        swap(xe, ye);
    }
    if (xb > xe) {  // 如果要从高到低绘制，等价于从低到高绘制
        swap(xb, xe);
        swap(yb, ye);
    }
    int dx = xe - xb, dy = abs(ye - yb),
            error = dx / 2, ystep = 1;
    if (y1 < y0) ystep = -1;
    for(int i = xb, j = yb;i<xe;i++){
        if(steep) p(i,j)=1;
        else p(j,i)=1;
        error-=dy;
        if(error<0){
            j+=ystep;
            error+=dx;
        }
    }
}

void draw_line_4(MatrixXf &p, int xb, int yb, int xe, int ye) {
    bool steep = abs(ye - yb) > abs(xe - xb);// 判断斜率是否大于1
    if (steep) {  // 如果斜率大于1，先计算镜像
        swap(xb, yb);
        swap(xe, ye);
    }
    if (xb > xe) {  // 如果要从高到低绘制，等价于从低到高绘制
        swap(xb, xe);
        swap(yb, ye);
    }
    int dx = xe - xb, dy = abs(ye - yb),
            px = 2*dy-dx, ystep = 1;
    if (y1 < y0) ystep = -1;

    for(int i = xb, j = yb;i<xe;i++){
        if(steep) p(i,j)=1;
        else p(j,i)=1;
//        error-=dy;
        if(px>0){
            j+=ystep;
            px+=(2*dy-2*dx);
        }
        else px+=(2*dy);
    }
}