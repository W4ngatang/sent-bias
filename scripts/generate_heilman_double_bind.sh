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
    local sentences_to_keep="$1"
    shift

    local name=
    for name in $names
    do
        field=`echo -en "\t$success_text" | \
            sed "s/NAME/$name/g" | \
            sed "s/CAP_POSS_PRON/$cap_poss_pron/g" | \
            sed "s/POSS_PRON/$poss_pron/g" | \
            sed "s/CAP_PRON/$cap_pron/g;s/PRON/$pron/g" | \
            cut -d . -f "$sentences_to_keep"`
        if [ "${field: -1}" != . ]
        then
            field="$field."
        fi
        echo -n "$field"
    done
    echo
}

print_evaluations() {
    local adjectives="$1"
    shift

    local adjective=
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

print_tests() {
    local success_text="$1"
    shift
    local target_1_name="$1"
    shift
    local target_1="$1"
    shift
    local target_2_name="$1"
    shift
    local target_2="$1"
    shift
    local sentences_to_keep="$1"
    shift

    print_header
    echo -n Female
    print_contexts "$success_text" "$female_names" Her her She she "$sentences_to_keep"
    echo -n Male
    print_contexts "$success_text" "$male_names" His his He he "$sentences_to_keep"
    echo -n "$target_1_name"
    print_evaluations "$target_1"
    echo -n "$target_2_name"
    print_evaluations "$target_2"
}

for sentences_to_keep in 1 1- 1,3-
do
    print_tests \
        "$ambig_success_text" \
        CompetentAchievementOriented "$competent" \
        IncompetentNotAchievementOriented "$incompetent" \
        $sentences_to_keep \
        > ../tests/heilman_double_bind_ambiguous_$sentences_to_keep.txt
    print_tests \
        "$clear_success_text" \
        LikableNotHostile "$likable" \
        UnlikableHostile "$unlikable" \
        $sentences_to_keep \
        > ../tests/heilman_double_bind_clear_$sentences_to_keep.txt
done
