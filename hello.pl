use strict;
use warnings;
use utf8;
use Text::CSV;

sub main{
    my $file = "C:/Users/jlu39/Desktop/jerry.txt";
    open(my $FH, '<', $file) or die ("File $file not found");
    my $counter = 0;
    while(my $line = <$FH>){
        my $sign = 0;
        if ( $line =~ /(\d+\.\d+\s?(°C))/g) {
            my $match = $1;
            my $num = (split /°C/,$match)[0];
            if($num >= 38){                                                 #febrile
                $sign = $sign || 1;
            }else{                                                          #not febrile
                $sign = $sign || 0;
            }
        }
        if($line =~ /(\d+\.?\d+?\s?(°F|F))/i){
            my $match = $1;
            my $num = (split /°F|f|F/,$match)[0];
            if($num >= 100.4){ 
                $sign = $sign || 1;
            }else{                                                          #not febrile
                $sign = $sign || 0;
            }
        }
        if($line =~ /(?:fever |fevers |temperature |temperatures |Tmax |Febrile )(?:\S+\s+){0,4}(\d{2,3}\.?\d?)/i){
            
            my $match = $1;
            if($match >= 100.4){
                $sign = $sign || 1;
            }
        }
        $counter++;

        print $sign;
       
        if($sign == 1){
            print $counter;
            print "\n";
        }
    }
    close($FH);
}
main();