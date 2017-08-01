function y = modu(x,dzdy)

  if nargin <= 1 || isempty(dzdy)
    y = abs(x) ;
  else
    y = dzdy .* sign(x) ;
  end
  
end
