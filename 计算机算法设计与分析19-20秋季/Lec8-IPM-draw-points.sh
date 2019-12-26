while read line; do x=`echo $line | cut -f 2 -d" "`; y=`echo $line | cut -f 3 -d " "`; echo "\\draw[red, fill=red] ($x, $y) circle (3pt);"; done < aaa
