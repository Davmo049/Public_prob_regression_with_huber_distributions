#ifndef __WELZL_H__
#define __WELZL_H__

struct welzl_circle
{
    double* p;
    double radius;
};

void welzl_plain(double* center, double* radius, double* points, int N, int d);
void welzl_iterative(struct welzl_circle* ret, double* points, int N, int d);
#endif
