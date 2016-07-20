// initialization of variables

#include <iostream>
using namespace std;

int main ()
{
  int a=5;               // initial value: 5
  int b(3);              // initial value: 3
  int c{2};              // initial value: 2
  int result;            // initial value undetermined
  double x = 5.7;
  a = a + b;
  result = a - c;
  long d;
  d = (long)(x);
  cout << d;

  return 0;
}