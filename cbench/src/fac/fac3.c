int
fac(int n)
{
  int f = 1;

  while (n)
    f *= n--;
  return f;
}

int
main()
{
  printf("%d\n", fac(5));
  return 0;
}
