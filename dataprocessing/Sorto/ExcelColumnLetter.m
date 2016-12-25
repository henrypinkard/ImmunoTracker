function y=ExcelColumnLetter(col)
% Convert column numbers to Excel style letters (maximum ZZZZ)
%
% A simple function to convert an integer input to the equivalent string
% for numbering columns in Microsoft Excel worksheets
%
% Example: str=ExcelColumnLetter(n)
%
% where:
%           n is the column number (limited to between 1 and 475254)
%           str is the equivalent Excel string
%
% If n==1,   str=='A'
%    n==27,  str=='AA' 
%    n=999,  str=='ALK' etc
% 
% NB Your version of Excel may limit the number of columns in a worksheet
% e.g. up to 2003 only 256 were permitted.
%
% -------------------------------------------------------------------------
% Author: Malcolm Lidierth 10/08
% -------------------------------------------------------------------------

if col<1 || col>26+26^2+26^3+26^4 || rem(col,1)~=0
    error('Column number must be whole number between 1 and %d', 26+26^2+26^3+26^4);
elseif col>=1 && col<=26
    y=rem(col-1, 26);
elseif col>26 && col<=26+26^2
    x=col-26-1;
    y=[x/26 rem(x,26)];
elseif col>26+26^2 && col<=26+26^2+26^3
    x=col-26^2-26-1;
    y=[x/26^2 rem(x/26,26) rem(x,26)];
elseif col>26+26^2+26^3 && col<=26+26^2+26^3+26^4
    x=col-26^3-26^2-26-1;
    y=[x/26^3 rem(x/26^2,26) rem(x/26,26) rem(x,26)];
end
y=char(y+65);

return
end