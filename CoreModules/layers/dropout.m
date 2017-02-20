function [y,mask] = dropout(x,dzdy,opts)

% determine mask
mask = opts.mask ;

if isempty(dzdy)
    scale = cast(1 / (1 - opts.rate), 'like', x) ;
    mask = scale * (rand(size(x), 'like', x) >= opts.rate) ;         
end

if isempty(dzdy)
    y = mask .* x ;   
else
    y = mask .* dzdy ;   
end
