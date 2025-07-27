int
main()
{
  int s, r, n = 5, u, v;

  for (u = r = 1; v = u, r < n; r++)
    for (s = 1; u += v, s++ < r; )
      ;
  printf("%d\n", u);
  return 0;
}
