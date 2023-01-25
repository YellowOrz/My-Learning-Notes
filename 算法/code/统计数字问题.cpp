//
// Created by orz on 2020/10/22.
//

#include <iostream>
#include <cmath>

using namespace std;

// 获取位数和最高位的数字
void get_topdigit(int a,int &digit,int &num){
    digit=0,num=a;
    while(num/10!= 0){
        digit++;
        num/=10;
    }
    digit++;
}

// 统计数字
void counter(long long pagenum,int count[]){
    int digit=0,top=0;
    get_topdigit(pagenum, digit, top);
    if(digit==1){
        for(int i = 1; i <= top; i++)
            count[i]++;
        return;
    }

    // 如果数字的位数≥2，怎把数字 Xxx...x 分成两部分，即拥有最高位的数字 X00...0 和低位的数 0xx...x
    long long part1= pow(10,digit-1) * top, part2= pagenum - part1;

    // 先从 X00...0 统计
    for(int i = 0;i<10;i++) //除去高位中，统计每个数字的个数：f(n)=x*f(n-1)=x*(n-1)*10^{n-2}
        count[i] += (top * (digit - 1) * pow(10, digit - 2));
    for(int i = 1; i < top; i++) //高位中，统计0~X-1数字的个数：f(n)=10^{n-1}
        count[i] += pow(10,digit-1);
    count[top]++;  //算上X00...0本身中数字的个数
    count[0]+=(digit-1);
    count[top] += part2;    // X对应数字的个数要加xx...x
    for(int i=0;i<digit-1;i++)  //最后剔除0开头的数字中0的个数：f(0)=\sum_{i=0}^{n-1-1}10^i
        count[0]-=pow(10,i);

    //统计 低位的数 xx...x
    counter(part2,count);
}

int main(){
    long long pagenum;
    cout << "Please input page number: ";
    cin >> pagenum;
    int count[10]={0};
    counter(pagenum,count);
    for(int i = 0;i<10;i++)
        cout << i<<"\t"<<count[i]<<endl;
    return 0;
}
