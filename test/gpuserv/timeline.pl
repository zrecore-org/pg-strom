#!/usr/bin/perl
# Copies stdin to stdout unchanged and records how long each line took to appear (the time since the previous line is
# charged to the line). A check line ("  ok    s1: ...") is charged to its prefix (the table or test group); setup output
# before it (table creation, NOTICEs) is charged to the next check. At the end the summary is written to the file given as
# the first argument: groups by seconds with ok/FAIL counts, then the slowest checks.
#
#   some_test 2>&1 | perl timeline.pl log.times > log
use strict;
use warnings;
use Time::HiRes qw(time);

my $out = shift or die "usage: timeline.pl <summary file>\n";
$| = 1;
my $t0 = time();
my $last = $t0;
my (@order, %sec, @lines);
my $header = '(start)';
my $pending = 0;        # seconds of setup lines, charged to the next check

sub bucket
{
	my ($name) = @_;
	if (!exists $sec{$name})
	{
		push @order, $name;
		$sec{$name} = { secs => 0, ok => 0, fail => 0 };
	}
	return $sec{$name};
}

while (my $l = <STDIN>)
{
	print $l;
	my $now = time();
	my $d = $now - $last;
	$last = $now;
	chomp(my $c = $l);
	if ($c =~ /^  (ok|FAIL)\s+(.*)$/)
	{
		my ($res, $rest) = ($1, $2);
		# a check is charged to its table or section prefix ("s1: ..."), else to the last header
		my $name = ($rest =~ /^([A-Za-z0-9_.]+):\s/) ? $1 : $header;
		my $b = bucket($name);
		$d += $pending;
		$pending = 0;
		$b->{secs} += $d;
		$b->{ok}++ if $res eq 'ok';
		$b->{fail}++ if $res eq 'FAIL';
		push @lines, [ $d, $c ];
	}
	else
	{
		# not a check: a header ("joins", "table s1") or setup output; its time belongs to the checks after it
		$header = $c if $c =~ /^[a-z]/ && $c !~ /^(table|notice|warning|hint|detail|context|set|drop|create|alter|insert|select|vacuum|analyze|update|delete|line)\b/i
			|| $c =~ /^table \S+$/;
		$pending += $d;
	}
}
bucket('(setup after the last check)')->{secs} += $pending if $pending > 0.5;
my $total = time() - $t0;
open(my $fh, '>', $out) or die "$out: $!\n";
printf $fh "total %.0f s\n\n%8s %5s %5s  section (prefix of the check names)\n", $total, 'seconds', 'ok', 'FAIL';
for my $s (sort { $sec{$b}{secs} <=> $sec{$a}{secs} } @order)
{
	my $r = $sec{$s};
	printf $fh "%8.1f %5d %5d  %s\n", $r->{secs}, $r->{ok}, $r->{fail}, $s;
}
printf $fh "\nslowest checks (setup before a check included)\n";
my $n = 0;
for my $e (sort { $b->[0] <=> $a->[0] } @lines)
{
	printf $fh "%8.1f  %s\n", $e->[0], substr($e->[1], 0, 150);
	last if ++$n >= 25;
}
close $fh;
