function [  ] = linechartwithuncertainty( x,y,err )

plot(x,y,'-','LineWidth',2);
hold on
patch([x fliplr(x)],[y+err fliplr(y-err)],[0.7 0.7 0.7], 'EdgeColor','none','FaceAlpha',0.5);
hold off

end

