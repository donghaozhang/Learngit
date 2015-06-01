/* rand example: guess the number */
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h> 
#include <iostream>

using namespace std;

#define PI 3.14159265

int main ()
{
  /* initialize random seed: */
  srand (time(NULL));
  //random ranges from 0 to 1
  float random = ((float) rand()) / (float) RAND_MAX;
  cout<<"Random between 0 and 1:"<<random<<endl;
  float normvector[3];
  //assign x, y, z direction value to the normal vector
  normvector[0] = 3;
  normvector[1] = 4;
  normvector[2] = 5;
  cout<<"normvector x direction:"<<normvector[0]<<"normvector y direction:"\
  <<normvector[1]<<"normvector z direction:"<<normvector[2]<<endl;

  float singleP[3]; float normvectorangle[2]; float radius; float t; float center[3]; 
  //assign theta phi value to the normal vector
  normvectorangle[0] = 0.3; normvectorangle[1] = 0.4;
  cout<<"normvector phi value:"<<normvectorangle[0]<<"normvector theta value:"<<normvectorangle[1]<<endl;

  double param, result;
  param = 60.0;
  result = cos ( param * PI / 180.0 );
  printf ("The cosine of %f degrees is %f.\n", param, result );
  singleP[0] = radius * cos(t) * (-sin(phi)) + radius * sin(t) * cos(theta) * cos(phi) +  center[1];
  singleP[1] = radius * cos(t) * cos(phi) + radius * sin(t) * cos(theta) * sin(phi) +  center[2];
  singleP[2] = radius * cos(t) * (-sin(phi)) + radius * sin(t) * cos(theta) * cos(phi) +  center[3];

  return 0;
}

