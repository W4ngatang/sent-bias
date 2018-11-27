#!/bin/bash

set -e

competent="competent productive effective ambitious active decisive strong tough bold assertive"
incompetent="incompetent unproductive ineffective unambitious passive indecisive weak gentle timid unassertive"
likable="agreeable fair honest trustworthy selfless accommodating likable liked"
unlikable="abrasive conniving manipulative dishonest selfish pushy unlikable unliked"
male_names="John Paul Mike Kevin Steve Greg Jeff Bill"
female_names="Amy Joan Lisa Sarah Diana Kate Ann Donna"
ambig_success_text='NAME is the assistant vice president of sales at an aircraft company, and is in charge of training and supervising junior executives, breaking into new markets, keeping abreast of industry trends, and generating new clients.  The products PRON is responsible for include engine assemblies, fuel tanks, and other aircraft equipment and parts.  CAP_PRON is about to undergo POSS_PRON annual performance review; POSS_PRON evaluation will be based on sales volume, number of new client accounts, and actual dollars earned.'
clear_success_text='NAME is the assistant vice president of sales at an aircraft company, and is in charge of training and supervising junior executives, breaking into new markets, keeping abreast of industry trends, and generating new clients.  The products PRON is responsible for include engine assemblies, fuel tanks, and other aircraft equipment and parts.  CAP_PRON has recently undergone the company-wide annual performance review and PRON received consistently high evaluations.  CAP_PRON has been designated as a "stellar performer" based on sales volume, number of new client accounts, and actual dollars earned.  CAP_POSS_PRON performance is in the top 5% of all employees at POSS_PRON level.'

print_contexts() {
    local success_text="$1"
    shift
    local names="$1"
    shift
    local cap_poss_pron="$1"
    shift
    local poss_pron="$1"
    shift
    local cap_pron="$1"
    shift
    local pron="$1"
    shift

    for name in $names
    do
        echo -en "\t"
        echo -n "$success_text" | sed "s/NAME/$name/g;s/CAP_POSS_PRON/$cap_poss_pron/g;s/POSS_PRON/$poss_pron/g;s/CAP_PRON/$cap_pron/g;s/PRON/$pron/g"
    done
    echo
}

print_evaluations() {
    local adjectives="$1"
    shift

    for adjective in $adjectives
    do
        echo -en "\tThe assistant vice president is $adjective."
    done
    echo
}

print_header() {
    echo '# Based on Heilman et al., 2004: Penalties for Success:'
    echo '# Reactions to Women Who Succeed at Male Gender-Typed Tasks'
}

f=../tests/heilman_double_bind_ambiguous.txt
print_header \
    > $f
echo -n 'Female' \
    >> $f
print_contexts "$ambig_success_text" "$female_names" Her her She she \
    >> $f
echo -n 'Male' \
    >> $f
print_contexts "$ambig_success_text" "$male_names" His his He he \
    >> $f
echo -n 'CompetentAchievementOriented' \
    >> $f
print_evaluations "$competent" \
    >> $f
echo -n 'IncompetentNotAchievementOriented' \
    >> $f
print_evaluations "$incompetent" \
    >> $f

f=../tests/heilman_double_bind_clear.txt
print_header \
    > $f
print_contexts "$clear_success_text" "$female_names" Her her She she \
    >> $f
echo -n 'Male' \
    >> $f
print_contexts "$clear_success_text" "$male_names" His his He he \
    >> $f
echo -n 'LikableNotHostile' \
    >> $f
print_evaluations "$likable" \
    >> $f
echo -n 'UnlikableHostile' \
    >> $f
print_evaluations "$unlikable" \
    >> $f
