use strict;
use warnings;
use utf8;
use Text::CSV_XS;
use Data::Dumper;

sub main{
    my $csv = Text::CSV_XS->new({binary=>1, auto_diag => 1}); #good practice acc. CPAN
    our $file = $ARGV[0] or die ("CSV $file not found");
    open(my $FH, '<', $file) or die;
    
        my @header = @{$csv->getline_all($FH)};
        my @encounter = map{$_ ->[9]} @header;                                  #column 9 
        my @notes = map{$_ ->[13]} @header;
        my @manual = map{$_ ->[2]} @header;
        my @mismatch;
        my @ctrFeButNeg;
        my @ctrAfButPos;
        my @trueFever;
        my @trueAf;
        my @empty;
        my %hash;
        
        my $size = 0;
        my $boolean = 0;                                                     #default as no fever                                                            #link encounter with boolean
        my $letter;
        my $prevIndex = 0;
        for(my $index = 0; $index <= (scalar @encounter)-2; $index++){                                                                            #prevent indexOutOfBound
            if($encounter[$index+1] == $encounter[$index]){    
                $boolean = $boolean || regex($notes[$index]);  
            }else{
                $hash{$encounter[$index]}{$manual[$prevIndex]} = $boolean;                    #new encounter id
                if(($manual[$prevIndex] eq 'Y' && $boolean == 0)){
                    push @ctrFeButNeg, $encounter[$index];
                    push @mismatch, $encounter[$index];
                }
                if( $manual[$prevIndex] eq 'N' && $boolean == 1){
                    push @ctrAfButPos, $encounter[$index];
                    push @mismatch, $encounter[$index];
                }
                if( $manual[$prevIndex] eq 'Y' && $boolean == 1){
                    push @trueFever, $encounter[$index];
                }
                if( $manual[$prevIndex] eq 'N' && $boolean == 0){
                    push @trueAf, $encounter[$index];
                }
                if( $manual[$prevIndex] eq ' '){
                    push @empty, $encounter[$index];
                }
                $boolean = 0;
                $prevIndex = $index+1;
                $size += 1;
            }
        }
        #sort keys %hash;
        print Dumper(\%hash);
        print Dumper(\@mismatch);
        my $rate = (1 - (scalar @mismatch) / $size)*100;
        print "sample of: $size\n";
        print "Correct rate: $rate %\n";
        print "     True\n";
        print "   ----------\n";
        print "T   "; print(scalar @trueFever);print "  | "; print (scalar @ctrFeButNeg);
        print "\nE  ---------\n";
        print "S   "; print (scalar @ctrAfButPos); print "  | "; print (scalar @trueAf);
        print "\nT\n";
        #print "With "; print (scalar @empty); print " entries";
        print "Afebrile but diagnosed as febrile are\n";
        print Dumper(\@ctrAfButPos);
        print "\nFebrile but diagnosed as Afebrile\n";
        print Dumper(\@ctrFeButNeg);
        $csv->eof or $csv->error_diag;   
    close($FH);
}

sub regex{
        my $line = $_[0];
        my $sign = 0;
        if($line =~/GOLISANO CHILDREN'S HOSPITAL/ || $line =~/After Visit Summary Signature Page/){
            return 0;
        }
        
        if ( $line =~ /(\d+\.\d+\s?(°C))/g) {
            my $match = $1;
            my $num = (split /°C/,$match)[0];
            if($num >= 38){                                                 #febrile
                $sign = $sign || 1;
            }else{                                                          #not febrile
                $sign = $sign || 0;
            }
        }
        if($line =~ /(\d+\.\d+\s?(°F|f|F))/i){
            my $match = $1;
            my $num = (split /°F|f|F/,$match)[0];
            if($num >= 100.4){ 
                $sign = $sign || 1;
            }else{                                                          #not febrile
                $sign = $sign || 0;
            }
        }
        if($line =~ /(?:fever |temperature |Tmax )(?:\S+\s+){1,4}(\d{2,3}\.?\d?)/){
            #print "sdfa";
            my $match = $1;
            if($match >= 100.4){
                $sign = $sign || 1;
            }
        }

        return $sign;
}

main();